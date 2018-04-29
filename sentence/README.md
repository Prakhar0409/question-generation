Model:
			----------           		------------
--sent--[+para]--->	| Encoder| ---[Latent Info]--> 	| Decoder  |   ----Generated Question---->
			----------           		------------

Steps:

* From the source (sentence) and target (question) in the training set, learn the vocabulary for source and target and save the training and validation data
	Some parameters to control size of src vocabulary etc and the training examples to consider (with src_seq_length smaller than 100)
	
	vocab.pt: stores the "a list of tuple (name, torchtext.Vocab)" - vocab of source and target from training data

`python preprocess.py -train_src ../data/processed/src-train.txt -train_tgt ../data/processed/tgt-train.txt  -valid_src ../data/processed/src-dev.txt -valid_tgt ../data/processed/tgt-dev.txt -save_data ./data/qg -src_vocab_size 45000 -tgt_vocab_size 28000 -src_seq_length 100`

* Convert the vocabulary into embeddings (initialize with something say glove.840B.300d.txt)

`./OpenNMT-py/tools/embeddings_to_torch.py -emb_file_enc ../../scratch/glove.840B.300d.txt -emb_file_dec ../../scratch/glove.840B.300d.txt -dict_file data/qg.vocab.pt -output_file data/embeddings`

	emb_file_enc: source Embeddings from this file 
	emb_file_dec: target Embeddings from this file 

* Train the model

`python train.py -data data/qg -rnn_type LSTM -encoder_type brnn -enc_layers 2 -dec_layers 2 -rnn_size 600 -dropout 0.3 -bridge -global_attention general -pre_word_vecs_enc data/embeddings.enc.pt -pre_word_vecs_dec data/embeddings.dec.pt -save_model snapshots/model-embs-lstm/ -word_vec_size 300 -gpuid 0`

* Translate the model

`python translate.py -model snapshots/model-embs-lstm/_acc_46.22_ppl_28.97_e13.pt -src ../data/processed/src-test.txt`

## preprocess.py [-h] [-md] [-data_type DATA_TYPE] -train_src TRAIN_SRC
##                     -train_tgt TRAIN_TGT -valid_src VALID_SRC -valid_tgt
#                     VALID_TGT [-src_dir SRC_DIR] -save_data SAVE_DATA
#                     [-max_shard_size MAX_SHARD_SIZE] [-src_vocab SRC_VOCAB]
#                     [-tgt_vocab TGT_VOCAB]
#                     [-features_vocabs_prefix FEATURES_VOCABS_PREFIX]
#                     [-src_vocab_size SRC_VOCAB_SIZE]
#                     [-tgt_vocab_size TGT_VOCAB_SIZE]
#                     [-src_words_min_frequency SRC_WORDS_MIN_FREQUENCY]
#                     [-tgt_words_min_frequency TGT_WORDS_MIN_FREQUENCY]
#                     [-dynamic_dict] [-share_vocab]
#                     [-src_seq_length SRC_SEQ_LENGTH]
#                     [-src_seq_length_trunc SRC_SEQ_LENGTH_TRUNC]
#                     [-tgt_seq_length TGT_SEQ_LENGTH]
#                     [-tgt_seq_length_trunc TGT_SEQ_LENGTH_TRUNC] [-lower]
#                     [-shuffle SHUFFLE] [-seed SEED]
#                     [-report_every REPORT_EVERY] [-sample_rate SAMPLE_RATE]
#                     [-window_size WINDOW_SIZE] [-window_stride WINDOW_STRIDE]
#                     [-window WINDOW]
#

python train.py -word_vec_size 300 -encoder_type brnn -decoder_type rnn -layers 2 -rnn_size 600 -input_feed 1 -rnn_type LSTM -brnn_merge sum -global_attention general  -data data/qg -save_model snapshots/try1/ -gpuid 0 -pre_word_vecs_enc data/embeddings.enc.pt -pre_word_vecs_dec data/embeddings.dec.pt -fix_word_vecs_enc 1 -fix_word_vecs_dec 1
usage: train.py [-h] [-md] [-src_word_vec_size SRC_WORD_VEC_SIZE]
                [-tgt_word_vec_size TGT_WORD_VEC_SIZE]
                [-word_vec_size WORD_VEC_SIZE] [-share_decoder_embeddings]
                [-share_embeddings] [-position_encoding]
                [-feat_merge {concat,sum,mlp}] [-feat_vec_size FEAT_VEC_SIZE]
                [-feat_vec_exponent FEAT_VEC_EXPONENT]
                [-model_type MODEL_TYPE]
                [-encoder_type {rnn,brnn,mean,transformer,cnn}]
                [-decoder_type {rnn,transformer,cnn}] [-layers LAYERS]
                [-enc_layers ENC_LAYERS] [-dec_layers DEC_LAYERS]
                [-rnn_size RNN_SIZE] [-cnn_kernel_width CNN_KERNEL_WIDTH]
                [-input_feed INPUT_FEED] [-bridge] [-rnn_type {LSTM,GRU,SRU}]
                [-brnn] [-brnn_merge {concat,sum}]
                [-context_gate {source,target,both}]
                [-global_attention {dot,general,mlp}] [-copy_attn]
                [-copy_attn_force] [-reuse_copy_attn]
                [-copy_loss_by_seqlength] [-coverage_attn]
                [-lambda_coverage LAMBDA_COVERAGE] -data DATA
                [-save_model SAVE_MODEL] [-gpuid GPUID [GPUID ...]]
                [-seed SEED] [-start_epoch START_EPOCH]
                [-param_init PARAM_INIT] [-param_init_glorot]
                [-train_from TRAIN_FROM]
                [-pre_word_vecs_enc PRE_WORD_VECS_ENC]
                [-pre_word_vecs_dec PRE_WORD_VECS_DEC] [-fix_word_vecs_enc]
                [-fix_word_vecs_dec] [-batch_size BATCH_SIZE]
                [-batch_type {sents,tokens}] [-normalization {sents,tokens}]
                [-accum_count ACCUM_COUNT]
                [-valid_batch_size VALID_BATCH_SIZE]
                [-max_generator_batches MAX_GENERATOR_BATCHES]
                [-epochs EPOCHS]
                [-optim {sgd,adagrad,adadelta,adam,sparseadam}]
                [-adagrad_accumulator_init ADAGRAD_ACCUMULATOR_INIT]
                [-max_grad_norm MAX_GRAD_NORM] [-dropout DROPOUT]
                [-truncated_decoder TRUNCATED_DECODER]
                [-adam_beta1 ADAM_BETA1] [-adam_beta2 ADAM_BETA2]
                [-label_smoothing LABEL_SMOOTHING]
                [-learning_rate LEARNING_RATE]
                [-learning_rate_decay LEARNING_RATE_DECAY]
                [-start_decay_at START_DECAY_AT]
                [-start_checkpoint_at START_CHECKPOINT_AT]
                [-decay_method {noam}] [-warmup_steps WARMUP_STEPS]
                [-report_every REPORT_EVERY] [-exp_host EXP_HOST] [-exp EXP]
                [-tensorboard] [-tensorboard_log_dir TENSORBOARD_LOG_DIR]
                [-sample_rate SAMPLE_RATE] [-window_size WINDOW_SIZE]

train.py

optional arguments:
  -h, --help            show this help message and exit
  -md                   print Markdown-formatted help text and exit.

Model-Embeddings:
  -src_word_vec_size SRC_WORD_VEC_SIZE
                        Word embedding size for src. (default: 500)
  -tgt_word_vec_size TGT_WORD_VEC_SIZE
                        Word embedding size for tgt. (default: 500)
  -word_vec_size WORD_VEC_SIZE
                        Word embedding size for src and tgt. (default: -1)
  -share_decoder_embeddings
                        Use a shared weight matrix for the input and output
                        word embeddings in the decoder. (default: False)
  -share_embeddings     Share the word embeddings between encoder and decoder.
                        Need to use shared dictionary for this option.
                        (default: False)
  -position_encoding    Use a sin to mark relative words positions. Necessary
                        for non-RNN style models. (default: False)

Model-Embedding Features:
  -feat_merge {concat,sum,mlp}
                        Merge action for incorporating features embeddings.
                        Options [concat|sum|mlp]. (default: concat)
  -feat_vec_size FEAT_VEC_SIZE
                        If specified, feature embedding sizes will be set to
                        this. Otherwise, feat_vec_exponent will be used.
                        (default: -1)
  -feat_vec_exponent FEAT_VEC_EXPONENT
                        If -feat_merge_size is not set, feature embedding
                        sizes will be set to N^feat_vec_exponent where N is
                        the number of values the feature takes. (default: 0.7)

Model- Encoder-Decoder:
  -model_type MODEL_TYPE
                        Type of source model to use. Allows the system to
                        incorporate non-text inputs. Options are
                        [text|img|audio]. (default: text)
  -encoder_type {rnn,brnn,mean,transformer,cnn}
                        Type of encoder layer to use. Non-RNN layers are
                        experimental. Options are
                        [rnn|brnn|mean|transformer|cnn]. (default: rnn)
  -decoder_type {rnn,transformer,cnn}
                        Type of decoder layer to use. Non-RNN layers are
                        experimental. Options are [rnn|transformer|cnn].
                        (default: rnn)
  -layers LAYERS        Number of layers in enc/dec. (default: -1)
  -enc_layers ENC_LAYERS
                        Number of layers in the encoder (default: 2)
  -dec_layers DEC_LAYERS
                        Number of layers in the decoder (default: 2)
  -rnn_size RNN_SIZE    Size of rnn hidden states (default: 500)
  -cnn_kernel_width CNN_KERNEL_WIDTH
                        Size of windows in the cnn, the kernel_size is
                        (cnn_kernel_width, 1) in conv layer (default: 3)
  -input_feed INPUT_FEED
                        Feed the context vector at each time step as
                        additional input (via concatenation with the word
                        embeddings) to the decoder. (default: 1)
  -bridge               Have an additional layer between the last encoder
                        state and the first decoder state (default: False)
  -rnn_type {LSTM,GRU,SRU}
                        The gate type to use in the RNNs (default: LSTM)
  -brnn                 Deprecated, use `encoder_type`. (default: None)
  -brnn_merge {concat,sum}
                        Merge action for the bidir hidden states (default:
                        concat)
  -context_gate {source,target,both}
                        Type of context gate to use. Do not select for no
                        context gate. (default: None)

Model- Attention:
  -global_attention {dot,general,mlp}
                        The attention type to use: dotprod or general (Luong)
                        or MLP (Bahdanau) (default: general)
  -copy_attn            Train copy attention layer. (default: False)
  -copy_attn_force      When available, train to copy. (default: False)
  -reuse_copy_attn      Reuse standard attention for copy (default: False)
  -copy_loss_by_seqlength
                        Divide copy loss by length of sequence (default:
                        False)
  -coverage_attn        Train a coverage attention layer. (default: False)
  -lambda_coverage LAMBDA_COVERAGE
                        Lambda value for coverage. (default: 1)

General:
  -data DATA            Path prefix to the ".train.pt" and ".valid.pt" file
                        path from preprocess.py (default: None)
  -save_model SAVE_MODEL
                        Model filename (the model will be saved as
                        <save_model>_epochN_PPL.pt where PPL is the validation
                        perplexity (default: model)
  -gpuid GPUID [GPUID ...]
                        Use CUDA on the listed devices. (default: [])
  -seed SEED            Random seed used for the experiments reproducibility.
                        (default: -1)

Initialization:
  -start_epoch START_EPOCH
                        The epoch from which to start (default: 1)
  -param_init PARAM_INIT
                        Parameters are initialized over uniform distribution
                        with support (-param_init, param_init). Use 0 to not
                        use initialization (default: 0.1)
  -param_init_glorot    Init parameters with xavier_uniform. Required for
                        transfomer. (default: False)
  -train_from TRAIN_FROM
                        If training from a checkpoint then this is the path to
                        the pretrained model's state_dict. (default: )
  -pre_word_vecs_enc PRE_WORD_VECS_ENC
                        If a valid path is specified, then this will load
                        pretrained word embeddings on the encoder side. See
                        README for specific formatting instructions. (default:
                        None)
  -pre_word_vecs_dec PRE_WORD_VECS_DEC
                        If a valid path is specified, then this will load
                        pretrained word embeddings on the decoder side. See
                        README for specific formatting instructions. (default:
                        None)
  -fix_word_vecs_enc    Fix word embeddings on the encoder side. (default:
                        False)
  -fix_word_vecs_dec    Fix word embeddings on the encoder side. (default:
                        False)

Optimization- Type:
  -batch_size BATCH_SIZE
                        Maximum batch size for training (default: 64)
  -batch_type {sents,tokens}
                        Batch grouping for batch_size. Standard is sents.
                        Tokens will do dynamic batching (default: sents)
  -normalization {sents,tokens}
                        Normalization method of the gradient. (default: sents)
  -accum_count ACCUM_COUNT
                        Accumulate gradient this many times. Approximately
                        equivalent to updating batch_size * accum_count
                        batches at once. Recommended for Transformer.
                        (default: 1)
  -valid_batch_size VALID_BATCH_SIZE
                        Maximum batch size for validation (default: 32)
  -max_generator_batches MAX_GENERATOR_BATCHES
                        Maximum batches of words in a sequence to run the
                        generator on in parallel. Higher is faster, but uses
                        more memory. (default: 32)
  -epochs EPOCHS        Number of training epochs (default: 13)
  -optim {sgd,adagrad,adadelta,adam,sparseadam}
                        Optimization method. (default: sgd)
  -adagrad_accumulator_init ADAGRAD_ACCUMULATOR_INIT
                        Initializes the accumulator values in adagrad. Mirrors
                        the initial_accumulator_value option in the tensorflow
                        adagrad (use 0.1 for their default). (default: 0)
  -max_grad_norm MAX_GRAD_NORM
                        If the norm of the gradient vector exceeds this,
                        renormalize it to have the norm equal to max_grad_norm
                        (default: 5)
  -dropout DROPOUT      Dropout probability; applied in LSTM stacks. (default:
                        0.3)
  -truncated_decoder TRUNCATED_DECODER
                        Truncated bptt. (default: 0)
  -adam_beta1 ADAM_BETA1
                        The beta1 parameter used by Adam. Almost without
                        exception a value of 0.9 is used in the literature,
                        seemingly giving good results, so we would discourage
                        changing this value from the default without due
                        consideration. (default: 0.9)
  -adam_beta2 ADAM_BETA2
                        The beta2 parameter used by Adam. Typically a value of
                        0.999 is recommended, as this is the value suggested
                        by the original paper describing Adam, and is also the
                        value adopted in other frameworks such as Tensorflow
                        and Kerras, i.e. see: https://www.tensorflow.org/api_d
                        ocs/python/tf/train/AdamOptimizer
                        https://keras.io/optimizers/ . Whereas recently the
                        paper "Attention is All You Need" suggested a value of
                        0.98 for beta2, this parameter may not work well for
                        normal models / default baselines. (default: 0.999)
  -label_smoothing LABEL_SMOOTHING
                        Label smoothing value epsilon. Probabilities of all
                        non-true labels will be smoothed by epsilon /
                        (vocab_size - 1). Set to zero to turn off label
                        smoothing. For more detailed information, see:
                        https://arxiv.org/abs/1512.00567 (default: 0.0)

Optimization- Rate:
  -learning_rate LEARNING_RATE
                        Starting learning rate. Recommended settings: sgd = 1,
                        adagrad = 0.1, adadelta = 1, adam = 0.001 (default:
                        1.0)
  -learning_rate_decay LEARNING_RATE_DECAY
                        If update_learning_rate, decay learning rate by this
                        much if (i) perplexity does not decrease on the
                        validation set or (ii) epoch has gone past
                        start_decay_at (default: 0.5)
  -start_decay_at START_DECAY_AT
                        Start decaying every epoch after and including this
                        epoch (default: 8)
  -start_checkpoint_at START_CHECKPOINT_AT
                        Start checkpointing every epoch after and including
                        this epoch (default: 0)
  -decay_method {noam}  Use a custom decay rate. (default: )
  -warmup_steps WARMUP_STEPS
                        Number of warmup steps for custom decay. (default:
                        4000)

Logging:
  -report_every REPORT_EVERY
                        Print stats at this interval. (default: 50)
  -exp_host EXP_HOST    Send logs to this crayon server. (default: )
  -exp EXP              Name of the experiment for logging. (default: )
  -tensorboard          Use tensorboardX for visualization during training.
                        Must have the library tensorboardX. (default: False)
  -tensorboard_log_dir TENSORBOARD_LOG_DIR
                        Log directory for Tensorboard. This is also the name
                        of the run. (default: runs/onmt)

Speech:
  -sample_rate SAMPLE_RATE
                        Sample rate. (default: 16000)
  -window_size WINDOW_SIZE
                        Window size for spectrogram in seconds. (default:
                        0.02)
