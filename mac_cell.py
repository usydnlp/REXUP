import collections
import numpy as np
import tensorflow as tf
import my_ops
from my_ops import MultiHeadAttention
from my_ops import Block
import ops
from config import config

MACCellTuple = collections.namedtuple("MACCellTuple", ("control", "memory"))

'''
The MAC cell.

Recurrent cell for multi-step reasoning. Presented in https://arxiv.org/abs/1803.03067.
The cell has recurrent control and memory states that interact with the question 
and knowledge base (image) respectively.

The hidden state structure is MACCellTuple(control, memory)

At each step the cell performs by calling to three subunits: control, read and write.

1. The Control Unit computes the control state by computing attention over the question words.
The control state represents the current reasoning operation the cell performs.

2. The Read Unit retrieves information from the knowledge base, given the control and previous
memory values, by computing 2-stages attention over the knowledge base.

3. The Write Unit integrates the retrieved information to the previous hidden memory state,
given the value of the control state, to perform the current reasoning operation.
'''


class MACCell(tf.compat.v1.nn.rnn_cell.RNNCell):
    '''Initialize the MAC cell.
    (Note that in the current version the cell is stateful -- 
    updating its own internals when being called) 
    
    Args:
        vecQuestions: the vector representation of the questions. 
        [batchSize, ctrlDim]

        questionWords: the question words embeddings. 
        [batchSize, questionLength, ctrlDim]

        questionCntxWords: the encoder outputs -- the "contextual" question words. 
        [batchSize, questionLength, ctrlDim]

        questionLengths: the length of each question.
        [batchSize]
        
        memoryDropout: dropout on the memory state (Tensor scalar).
        readDropout: dropout inside the read unit (Tensor scalar).
        writeDropout: dropout on the new information that gets into the write unit (Tensor scalar).
        
        batchSize: batch size (Tensor scalar).
        train: train or test mod (Tensor boolean).
        reuse: reuse cell

        knowledgeBase: 
    '''

    def __init__(self, vecQuestions, questionWords, questionCntxWords, questionLengths,
                 knowledgeBase, memoryDropout, readDropout,
                 writeDropout, controlDropoutPre, controlDropoutPost, wordDropout, vocabDropout,
                 objectDropout, batchSize, train, kbSize=None, reuse=None, isTransformer=False, isBlock=False,
                 kbDim=2048, scopeName="", seqlen=100):

        self.vecQuestions = vecQuestions
        self.questionWords = questionWords
        self.questionCntxWords = questionCntxWords
        self.questionLengths = questionLengths

        self.knowledgeBase = knowledgeBase
        self.kbSize = kbSize

        self.dropouts = {}
        self.dropouts["memory"] = memoryDropout
        self.dropouts["read"] = readDropout
        self.dropouts["write"] = writeDropout
        self.dropouts["controlPre"] = controlDropoutPre
        self.dropouts["controlPost"] = controlDropoutPost
        self.dropouts["word"] = wordDropout
        self.dropouts["vocab"] = vocabDropout
        self.dropouts["object"] = objectDropout

        self.none = tf.zeros((batchSize, 1), dtype=tf.float32)

        self.batchSize = batchSize
        self.train = train
        self.reuse = reuse

        self.memEncoder = my_ops.MemEncoder(config.memDim, 8, 2048)
        self.mha_contrl = MultiHeadAttention(512, 8)
        self.memSameSizeWithKB = False
        self.block_fusion1 = Block(512,
                                   mm_dim=1600,
                                   chunks=20,
                                   rank=15,
                                   shared=False,
                                   dropout_input=0.1,
                                   dropout_pre_lin=0.,
                                   dropout_output=0.)
        self.block_fusion2 = Block(512,
                                   mm_dim=1600,
                                   chunks=20,
                                   rank=15,
                                   shared=False,
                                   dropout_input=0.1,
                                   dropout_pre_lin=0.,
                                   dropout_output=0.)

        self.isTransformer = isTransformer
        self.isBlock = isBlock
        self.kbDim = kbDim
        self.scopeName = scopeName
        self.seqlen = seqlen

    ''' 
    Cell state size. 
    '''

    @property
    def state_size(self):
        return MACCellTuple(config.ctrlDim, config.memDim)

    '''
    Cell output size. Currently it doesn't have any outputs. 
    '''

    @property
    def output_size(self):
        return 1

    '''
    The Control Unit: computes the new control state -- the reasoning operation,
    by summing up the word embeddings according to a computed attention distribution.
    
    The unit is recurrent: it receives the whole question and the previous control state,
    merge them together (resulting in the "continuous control"), and then uses that 
    to compute attentions over the question words. Finally, it combines the words 
    together according to the attention distribution to get the new control state. 
    
    Args:
        controlInput: external inputs to control unit (the question vector).
        [batchSize, ctrlDim]

        inWords: the representation of the words used to compute the attention.
        [batchSize, questionLength, ctrlDim]

        outWords: the representation of the words that are summed up. 
                  (by default inWords == outWords)
        [batchSize, questionLength, ctrlDim]

        questionLengths: the length of each question.
        [batchSize]

        control: the previous control hidden state value.
        [batchSize, ctrlDim]

        contControl: optional corresponding continuous control state
        (before casting the attention over the words).
        [batchSize, ctrlDim]

    Returns:
        the new control state
        [batchSize, ctrlDim]

        the continuous (pre-attention) control
        [batchSize, ctrlDim]
    '''

    def control(self, controlInput, inWords, outWords, questionLengths,
                append=True, control=None, contControl=None, name="", reuse=None):

        with tf.variable_scope("control" + name, reuse=reuse):
            dim = config.ctrlDim

            ## Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = controlInput
            if config.controlFeedPrev:
                newContControl = control if config.controlFeedPrevAtt else contControl
                if config.controlFeedInputs:
                    newContControl = tf.concat([newContControl, controlInput], axis=-1)
                    dim += config.ctrlDim

                # merge inputs together
                newContControl = ops.linear(newContControl, dim, config.ctrlDim,
                                            act=config.controlContAct, name="contControl")
                dim = config.ctrlDim

            ## Step 2: compute attention distribution over words and sum them up accordingly.
            # compute interactions with question words
            interactions = tf.expand_dims(newContControl, axis=1) * inWords

            # optionally concatenate words
            if config.controlConcatWords:
                interactions = tf.concat([interactions, inWords], axis=-1)
                dim += config.ctrlDim

                # optional projection
            if config.controlProj:
                interactions = ops.linear(interactions, dim, config.ctrlDim,
                                          act=config.controlProjAct)
                dim = config.ctrlDim

            # compute attention distribution over words and summarize them accordingly
            logits = ops.inter2logits(interactions, dim)

            if config.wordDp < 1.0:
                logits = tf.nn.dropout(logits, self.dropouts["word"])

            attention = tf.nn.softmax(ops.expMask(logits, questionLengths))
            if append:
                self.attentions["question"].append(attention)

            newControl = ops.att2Smry(attention, outWords)

            gateDim = config.ctrlDim

            # ablation: use continuous control (pre-attention) instead
            if config.controlContinuous:
                newControl = newContControl

        return newControl, newContControl

    def control_new(self, controlInput, inWords, outWords, questionLengths,
                    append=True, control=None, contControl=None, name="", reuse=None):

        with tf.variable_scope("control" + name, reuse=reuse):
            dim = config.ctrlDim

            ## Step 1: compute "continuous" control state given previous control and question.
            # control inputs: question and previous control
            newContControl = controlInput
            if config.controlFeedPrev:
                newContControl = control if config.controlFeedPrevAtt else contControl
                if config.controlFeedInputs:
                    newContControl = tf.concat([newContControl, controlInput], axis=-1)
                    dim += config.ctrlDim

                # merge inputs together
                newContControl = ops.linear(newContControl, dim, config.ctrlDim,
                                            act=config.controlContAct, name="contControl")
                dim = config.ctrlDim

            ## Step 2: compute attention distribution over words and sum them up accordingly.
            # compute interactions with question words
            interactions = tf.expand_dims(newContControl, axis=1) * inWords

            # optionally concatenate words
            if config.controlConcatWords:
                interactions = tf.concat([interactions, inWords], axis=-1)
                dim += config.ctrlDim

                # optional projection
            if config.controlProj:
                interactions = ops.linear(interactions, dim, config.ctrlDim,
                                          act=config.controlProjAct)
                dim = config.ctrlDim

            # compute attention distribution over words and summarize them accordingly 
            logits = ops.inter2logits(interactions, dim)

            if config.wordDp < 1.0:
                logits = tf.nn.dropout(logits, self.dropouts["word"])

            attention = tf.nn.softmax(ops.expMask(logits, questionLengths))
            if append:
                self.attentions["question"].append(attention)

            newControl = ops.att2Smry(attention, outWords)

            attend_words = tf.expand_dims(attention, axis=-1) * outWords

            gateDim = config.ctrlDim

            # ablation: use continuous control (pre-attention) instead
            if config.controlContinuous:
                newControl = newContControl

        return newControl, newContControl, attend_words

    '''
    The read unit extracts relevant information from the knowledge base given the
    cell's memory and control states. It computes attention distribution over
    the knowledge base by comparing it first to the memory and then to the control.
    Finally, it uses the attention distribution to sum up the knowledge base accordingly,
    resulting in an extraction of relevant information. 

    Args:
        knowledge base: representation of the knowledge base (image). 
        [batchSize, kbSize (Height * Width), memDim]

        memory: the cell's memory state
        [batchSize, memDim]

        control: the cell's control state
        [batchSize, ctrlDim]

    Returns the information extracted.
    [batchSize, memDim]
    '''

    def read(self, knowledgeBase, memory, control, name="", reuse=None):
        with tf.variable_scope("read" + name, reuse=reuse):
            dim = config.memDim
            interDim = dim

            ## memory dropout
            if config.memoryVariationalDropout:
                memory = ops.applyVarDpMask(memory, self.memDpMask, self.dropouts["memory"])
            else:
                memory = tf.nn.dropout(memory, self.dropouts["memory"])

            ## Step 1: knowledge base / memory interactions
            # parameters for knowledge base and memory projection
            proj = None
            if config.readProjInputs:
                proj = {"dim": config.attDim, "shared": config.readProjShared, "dropout": self.dropouts["read"]}
                dim = config.attDim

            # parameters for concatenating knowledge base elements
            concat = {"x": config.readMemConcatKB, "proj": config.readMemConcatProj}

            # compute interactions between knowledge base and memory
            interactions, interDim = ops.mul(x=knowledgeBase, y=memory, dim=config.memDim,
                                             proj=proj, concat=concat, interMod=config.readMemAttType, name="memInter")

            projectedKB = proj.get("x") if proj else None

            # project memory interactions back to hidden dimension
            if config.readMemProj:
                interactions = ops.linear(interactions, interDim, dim, act=config.readMemAct,
                                          name="memKbProj")
            else:
                dim = interDim

            ## Step 2: compute interactions with control
            if config.readCtrl:
                # compute interactions with control
                if config.ctrlDim != dim:
                    control = ops.linear(control, config.ctrlDim, dim, name="ctrlProj")

                interactions, interDim = ops.mul(interactions, control, dim,
                                                 interMod=config.readCtrlAttType, concat={"x": config.readCtrlConcatInter},
                                                 name="ctrlInter")

                # optionally concatenate knowledge base elements
                if config.readCtrlConcatKB:
                    if config.readCtrlConcatProj:
                        addedInp, addedDim = projectedKB, config.attDim
                    else:
                        addedInp, addedDim = knowledgeBase, config.memDim
                    interactions = tf.concat([interactions, addedInp], axis=-1)
                    dim += addedDim

                    # optional nonlinearity
                interactions = ops.activations[config.readCtrlAct](interactions)

            ## Step 3: sum attentions up over the knowledge base
            # transform vectors to attention distribution
            if config.objectDp < 1.0:
                interactions = tf.nn.dropout(interactions, self.dropouts["object"])

            attention = ops.inter2att(interactions, dim, dropout=self.dropouts["read"],
                                      mask=self.kbSize)

            self.attentions["kb"].append(attention)

            # optionally use projected knowledge base instead of original
            if config.readSmryKBProj:
                knowledgeBase = projectedKB

            # sum up the knowledge base according to the distribution
            information = ops.att2Smry(attention, knowledgeBase)

        return information

    def read_new(self, knowledgeBase, memory, control, name="", reuse=None):
        with tf.variable_scope("read" + name, reuse=reuse):
            dim = config.memDim
            interDim = dim

            ## memory dropout
            if config.memoryVariationalDropout:
                memory = ops.applyVarDpMask(memory, self.memDpMask, self.dropouts["memory"])
            else:
                memory = tf.nn.dropout(memory, self.dropouts["memory"])

            ## Step 1: knowledge base / memory interactions 
            # parameters for knowledge base and memory projection 
            proj = None
            if config.readProjInputs:
                proj = {"dim": config.attDim, "shared": config.readProjShared, "dropout": self.dropouts["read"]}
                dim = config.attDim

            # parameters for concatenating knowledge base elements
            concat = {"x": config.readMemConcatKB, "proj": config.readMemConcatProj}

            # compute interactions between knowledge base and memory
            interactions, interDim = ops.mul(x=knowledgeBase, y=memory, dim=config.memDim,
                                             proj=proj, concat=concat, interMod=config.readMemAttType, name="memInter")

            projectedKB = proj.get("x") if proj else None

            # project memory interactions back to hidden dimension
            if config.readMemProj and interDim != dim:
                interactions = ops.linear(interactions, interDim, dim, act=config.readMemAct,
                                          name="memKbProj")
            else:
                dim = interDim

            ## Step 2: compute interactions with control
            if config.readCtrl:
                # compute interactions with control
                if config.ctrlDim != dim:
                    control = ops.linear(control, config.ctrlDim, dim, name="ctrlProj")

                interactions = self.block_fusion2([interactions, tf.expand_dims(control, axis=1)], training=self.train, seqlen=self.seqlen)
               
                if config.readCtrlConcatKB:
                    if config.readCtrlConcatProj:
                        addedInp, addedDim = projectedKB, config.attDim
                    else:
                        addedInp, addedDim = knowledgeBase, config.memDim
                    interactions = tf.concat([interactions, addedInp], axis=-1)
                    dim += addedDim

                    # optional nonlinearity
                interactions = ops.activations[config.readCtrlAct](interactions)

            ## Step 3: sum attentions up over the knowledge base
            # transform vectors to attention distribution
            if config.objectDp < 1.0:
                interactions = tf.nn.dropout(interactions, self.dropouts["object"])

            attention = ops.inter2att(interactions, dim, dropout=self.dropouts["read"], mask=self.kbSize)
            # attention = ops.inter2att(control, dim, dropout=self.dropouts["read"], mask=self.kbSize)
            self.attentions["kb"].append(attention)

            # optionally use projected knowledge base instead of original
            if config.readSmryKBProj:
                knowledgeBase = projectedKB

            # sum up the knowledge base according to the distribution
            information = ops.att2Smry(attention, knowledgeBase)

        return information

    '''
    The write unit integrates newly retrieved information (from the read unit),
    with the cell's previous memory hidden state, resulting in a new memory value.
    The unit optionally supports:
    1. Self-attention to previous control / memory states, in order to consider previous steps
    in the reasoning process.
    2. Gating between the new memory and previous memory states, to allow dynamic adjustment
    of the reasoning process length.

    Args:
        memory: the cell's memory state
        [batchSize, memDim]

        info: the information to integrate with the memory
        [batchSize, memDim]

        control: the cell's control state
        [batchSize, ctrlDim]

        contControl: optional corresponding continuous control state 
        (before casting the attention over the words).
        [batchSize, ctrlDim]

    Return the new memory 
    [batchSize, memDim]
    '''

    def write(self, memory, info, control, contControl=None, name="", reuse=None):
        with tf.variable_scope("write" + name, reuse=reuse):

            # optionally project info
            if config.writeInfoProj:
                info = ops.linear(info, config.memDim, config.memDim, name="info")

            # optional info nonlinearity
            info = ops.activations[config.writeInfoAct](info)

            # compute self-attention vector based on previous controls and memories
            if config.writeSelfAtt:
                selfControl = control
                if config.writeSelfAttMod == "CONT":
                    selfControl = contControl
                selfControl = ops.linear(selfControl, config.ctrlDim, config.ctrlDim, name="ctrlProj")

                interactions = self.controls * tf.expand_dims(selfControl, axis=1)

                attention = ops.inter2att(interactions, config.ctrlDim, name="selfAttention")
                self.attentions["self"].append(attention)
                selfSmry = ops.att2Smry(attention, self.memories)

            # get write unit inputs: previous memory, the new info, optionally self-attention / control

            newMemory, dim = memory, config.memDim
            if config.writeInputs == "INFO":
                newMemory = info
            elif config.writeInputs == "SUM":
                newMemory += info
            elif config.writeInputs == "BOTH":
                newMemory, dim = ops.concat(newMemory, info, dim, mul=config.writeConcatMul)

            if config.writeSelfAtt:
                newMemory = tf.concat([newMemory, selfSmry], axis=-1)
                dim += config.memDim

            if config.writeMergeCtrl:
                # project memory back to memory dimension
                if config.writeMemProj or (dim != config.memDim):
                    newMemory = ops.linear(newMemory, dim, config.memDim, name="newMemoryCtrl")
                    dim = config.memDim

                if config.writeMergeCtrlMul:
                    newMemory = tf.concat([newMemory, newMemory * control], axis=-1)  # control,
                    dim += config.memDim  # 2 *
                else:
                    newMemory = tf.concat([newMemory, control], axis=-1)
                    dim += config.memDim

                    # project memory back to memory dimension
            if config.writeMemProj or (dim != config.memDim):
                newMemory = ops.linear(newMemory, dim, config.memDim, name="newMemory")

            # optional memory nonlinearity
            newMemory = ops.activations[config.writeMemAct](newMemory)

            # write unit gate
            if config.writeGate:
                gateDim = config.memDim
                if config.writeGateShared:
                    gateDim = 1

                z = tf.sigmoid(ops.linear(control, config.ctrlDim, gateDim, name="gate", bias=config.writeGateBias))

                self.attentions["gate"].append(z)

                newMemory = newMemory * z + memory * (1 - z)

                # optional batch normalization
            if config.memoryBN:
                newMemory = tf.contrib.layers.batch_norm(newMemory, decay=config.bnDecay,
                                                         center=config.bnCenter, scale=config.bnScale,
                                                         is_training=self.train, updates_collections=None)

        return newMemory

    def EncodeMem(self, memory, control, attend_words, knowledge, name="", reuse=None):
        with tf.variable_scope("memEncoder" + name, reuse=reuse):
            newControl_expand = tf.expand_dims(control, 1)
            if not self.memSameSizeWithKB:
                memory = tf.expand_dims(memory, 1)

            newMemory = self.memEncoder(memory, newControl_expand, attend_words, knowledge, self.train, None, self.kbSize)
            if not self.memSameSizeWithKB:
                newMemory = tf.reshape(newMemory, (-1, newMemory.shape[2]))

        return newMemory

    '''
    Call the cell to get new control and memory states.

    Args:
        inputs: in the current implementation the cell don't get recurrent inputs
        every iteration (argument for comparability with rnn interface).
            
        state: the cell current state (control, memory)
        MACCellTuple([batchSize, ctrlDim],[batchSize, memDim])

    Returns the new state -- the new memory and control values.
    MACCellTuple([batchSize, ctrlDim],[batchSize, memDim])
    '''

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__
        print("##### REXUP cell name:", self.scopeName + scope)
        with tf.variable_scope(self.scopeName + scope, reuse=self.reuse):  # as tfscope
            control = state.control
            memory = state.memory

            # cell sharing
            inputName = "qInput"
            inputNameU = "qInputU"
            inputReuseU = inputReuse = (self.iteration > 0)
            if config.controlInputUnshared:
                inputNameU = "qInput%d" % self.iteration
                inputReuseU = None

            cellName = ""
            cellReuse = (self.iteration > 0)
            if config.unsharedCells:
                cellName = str(self.iteration)
                cellReuse = None

                ## control unit
            # prepare question input to control 
            controlInput = ops.linear(self.vecQuestions, config.ctrlDim, config.ctrlDim,
                                      name=inputName, reuse=inputReuse)

            if not config.linearControl:
                controlInput = ops.activations[config.controlInputAct](controlInput)

                controlInput = ops.linear(controlInput, config.ctrlDim, config.ctrlDim,
                                          name=inputNameU, reuse=inputReuseU)

            if config.controlInWordsProj and config.controlOutWordsProj:
                if config.controlPostDropout < 1.0:
                    inWords = tf.nn.dropout(self.inWords, self.dropouts["controlPost"])
                    outWords = inWords
                else:
                    inWords = self.inWords
                    outWords = self.outWords
            else:
                inWords = self.inWords
                outWords = self.outWords

            if self.isTransformer:
                newControl, self.contControl, attend_words = self.control_new(controlInput, inWords, outWords,
                                                                              self.questionLengths, control=control, contControl=self.contControl, name=cellName, reuse=cellReuse)
            else:
                newControl, self.contControl = self.control(controlInput, inWords, outWords,
                                                            self.questionLengths, control=control, contControl=self.contControl, name=cellName, reuse=cellReuse)

            if self.isTransformer:
                newMemory = self.EncodeMem(memory, newControl, attend_words, self.knowledgeBase, name=cellName, reuse=cellReuse)

                self.infos = tf.concat([self.infos, tf.expand_dims(newMemory, axis=1)], axis=1)
            else:
                # read unit
                # ablation: use whole question as control
                if config.controlWholeQ:
                    newControl = self.vecQuestions


                if self.isBlock:
                    # read unit with BLOCK fusion
                    info = self.read_new(self.knowledgeBase, memory, newControl, name=cellName, reuse=cellReuse)
                else:
                    info = self.read(self.knowledgeBase, memory, newControl, name=cellName, reuse=cellReuse)

                if config.writeDropout < 1.0:
                    # write unit
                    info = tf.nn.dropout(info, self.dropouts["write"])

                newMemory = self.write(memory, info, newControl, self.contControl, name=cellName, reuse=cellReuse)
                self.infos = tf.concat([self.infos, tf.expand_dims(info, axis=1)], axis=1)

            # append as standard list?
            self.controls = tf.concat([self.controls, tf.expand_dims(newControl, axis=1)], axis=1)
            self.memories = tf.concat([self.memories, tf.expand_dims(newMemory, axis=1)], axis=1)

        newState = MACCellTuple(newControl, newMemory)
        return self.none, newState

    '''
    Initializes the a hidden state to based on the value of the initType:
    "PRM" for parametric initialization
    "ZERO" for zero initialization  
    "Q" to initialize to question vectors.

    Args:
        name: the state variable name.
        dim: the dimension of the state.
        initType: the type of the initialization
        batchSize: the batch size

    Returns the initialized hidden state.
    '''

    def initState(self, name, dim, initType, batchSize):
        if initType == "PRM":
            prm = tf.get_variable(name, shape=(dim,),
                                  initializer=tf.random_normal_initializer())
            initState = tf.tile(tf.expand_dims(prm, axis=0), [batchSize, 1])
        elif initType == "ZERO":
            initState = tf.zeros((batchSize, dim), dtype=tf.float32)
        else:  # "Q"
            initState = self.vecQuestions
        return initState

    def initmemState(self, name, dim, initType, batchSize):
        if initType == "PRM":
            prm = tf.get_variable(name, shape=dim,
                                  initializer=tf.random_normal_initializer())
            initState = tf.tile(tf.expand_dims(prm, axis=0), [batchSize, 1, 1])
        elif initType == "ZERO":
            initState = tf.zeros((batchSize, dim), dtype=tf.float32)
        else:  # "Q"
            initState = self.vecQuestions
        return initState

    '''
    Add a parametric null word to the questions.

    Args:
        words: the words to add a null word to.
        [batchSize, questionLentgth]

        lengths: question lengths.
        [batchSize] 

    Returns the updated word sequence and lengths.  
    '''

    def addNullWord(self, words, lengths):
        nullWord = tf.get_variable("zeroWord", shape=(1, config.ctrlDim), initializer=tf.random_normal_initializer())
        nullWord = tf.tile(tf.expand_dims(nullWord, axis=0), [self.batchSize, 1, 1])
        words = tf.concat([nullWord, words], axis=1)
        lengths += 1
        return words, lengths

    '''
    Initializes the cell internal state (currently it's stateful). In particular,
    1. Data-structures (lists of attention maps and accumulated losses).
    2. The memory and control states.
    3. The knowledge base (optionally merging it with the question vectors)
    4. The question words used by the cell (either the original word embeddings, or the 
       encoder outputs, with optional projection).

    Args:
        batchSize: the batch size

    Returns the initial cell state.
    '''

    def zero_state(self, batchSize, dtype=tf.float32):
        ## initialize data-structures
        self.attentions = {"kb": [], "question": [], "self": [], "gate": []}
        self.autoEncLosses = {"control": tf.constant(0.0), "memory": tf.constant(0.0)}

        ## initialize state
        initialControl = self.initState("initCtrl", config.ctrlDim, config.initCtrl, batchSize)
        if self.memSameSizeWithKB:
            initialMemory = self.initmemState("initMem", (100, config.memDim), config.initMem, batchSize)
        else:
            initialMemory = self.initState("initMem", config.memDim, config.initMem, batchSize)

        self.controls = tf.expand_dims(initialControl, axis=1)
        self.memories = tf.expand_dims(initialMemory, axis=1)
        self.infos = tf.expand_dims(initialMemory, axis=1)

        self.contControl = initialControl

        ## initialize knowledge base
        # optionally merge question into knowledge base representation
        if config.initKBwithQ != "NON":
            if config.imageEnsembleFeatures:
                self.knowledgeBase = ops.linear(self.knowledgeBase, self.kbDim, config.memDim, name="initKB")
            elif config.imageObjects:
                self.knowledgeBase = ops.linear(self.knowledgeBase, config.imageDims[-1], config.memDim, name="initKB")
            elif config.imageObjectsAndGrid:
                print("self.knowledgeBase", self.knowledgeBase.shape, "config.imageDims", config.imageDims)
                self.knowledgeBase = ops.linear(self.knowledgeBase, config.imageDims[-1] + 4, config.memDim, name="initKB")
            elif config.imageSceneGraph:
                print("self.knowledgeBase", self.knowledgeBase.shape, "config.imageDims", config.imageDims)
                self.knowledgeBase = ops.linear(self.knowledgeBase, 900, config.memDim, name="initKB")
            else:
                iVecQuestions = ops.linear(self.vecQuestions, config.ctrlDim, config.memDim, name="questions")

                concatMul = (config.initKBwithQ == "MUL")
                cnct, dim = ops.concat(self.knowledgeBase, iVecQuestions, config.memDim, mul=concatMul, expandY=True)
                self.knowledgeBase = ops.linear(cnct, dim, config.memDim, name="initKB")

        ## initialize question words
        # choose question words to work with (original embeddings or encoder outputs)
        words = self.questionCntxWords if config.controlContextual else self.questionWords

        # optionally add parametric "null" word in the to all questions
        if config.addNullWord:
            words, self.questionLengths = self.addNullWord(words, self.questionLengths)

        # project words
        if config.controlPreDropout < 1.0:
            words = tf.nn.dropout(words, self.dropouts["controlPre"])

        self.inWords = self.outWords = words
        if config.controlInWordsProj or config.controlOutWordsProj:
            pWords = ops.linear(words, config.wrdQEmbDim, config.ctrlDim, name="wordsProj")
            self.inWords = pWords if config.controlInWordsProj else words
            self.outWords = pWords if config.controlOutWordsProj else words

        ## initialize memory variational dropout mask
        if config.memoryVariationalDropout:
            if self.memSameSizeWithKB:
                self.memDpMask = ops.generateVarDpMask((batchSize, 100, config.memDim), self.dropouts["memory"])
            else:
                self.memDpMask = ops.generateVarDpMask((batchSize, config.memDim), self.dropouts["memory"])

        return MACCellTuple(initialControl, initialMemory)
