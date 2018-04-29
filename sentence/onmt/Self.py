import onmt

def buildInputNetwork(opt, vocab, pretrainedWords, fixWords):
  wordEmbedding = onmt.modules.Embeddings(len(vocab), opt.word_vec_size, pretrainedWords, fixWords)

  inputSize = opt.word_vec_size
  inputNetwork = wordEmbedding

  return inputNetwork, inputSize

def buildEncoder(opt, vocab):
  inputNetwork, inputSize = buildInputNetwork(opt, vocab, opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

  if opt.brnn:
    # if bidirectional
    rnnSize = opt.rnn_size
    if opt.brnn_merge == 'concat':
      # merger the two hidden states by concat
      if opt.rnn_size % 2 != 0:
        print('in concat mode, rnn_size must be divisible by 2')
      rnnSize = int(rnnSize / 2)
    elif opt.brnn_merge == 'sum':
      rnnSize = rnnSize
    else:
      print('invalid merge action ' + str(opt.brnn_merge))

    rnn = RNNEncoder('LSTM', opt.brnn, opt.layers, hidden_size=rnnSize, dropout=opt.dropout, embeddings=inputNetwork, use_bridge=False)
    #rnn = torch.LSTM.new(opt.layers, inputSize, rnnSize, opt.dropout, opt.residual)
    #return onmt.BiEncoder.new(inputNetwork, rnn, opt.brnn_merge)
    return rnn

def buildDecoder(opt, dicts, verbose):
  inputNetwork, inputSize = buildInputNetwork(opt, vocab, opt.pre_word_vecs_enc, opt.fix_word_vecs_enc)

"""#  generator = onmt.Generator.new(opt.rnn_size, dicts.words:size())

  if opt.input_feed == 1 then
    if verbose then
      print(" * using input feeding")
    end
    inputSize = inputSize + opt.rnn_size
  end

  local rnn = onmt.LSTM.new(opt.layers, inputSize, opt.rnn_size, opt.dropout, opt.residual)

  return onmt.Decoder.new(inputNetwork, rnn, generator, opt.input_feed == 1)
end
"""
