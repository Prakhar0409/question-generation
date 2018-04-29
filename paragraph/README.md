python preprocess.py -train_src ../data/processed/src-train.txt -train_tgt ../data/processed/tgt-train.txt -train_par ../data/processed/par-train.txt -valid_src ../data/processed/src-dev.txt -valid_tgt ../data/processed/tgt-dev.txt -valid_par ../data/processed/par-dev.txt -save_data ./data/qg -src_vocab_size 45000 -tgt_vocab_size 28000 -par_vocab_size 45000 -src_seq_length 100 -par_seq_length 150


preprocess.py [-h] [-md] [-data_type DATA_TYPE] -train_src TRAIN_SRC
                     -train_tgt TRAIN_TGT -valid_src VALID_SRC -valid_tgt
                     VALID_TGT [-src_dir SRC_DIR] -save_data SAVE_DATA
                     [-max_shard_size MAX_SHARD_SIZE] [-src_vocab SRC_VOCAB]
                     [-tgt_vocab TGT_VOCAB]
                     [-features_vocabs_prefix FEATURES_VOCABS_PREFIX]
                     [-src_vocab_size SRC_VOCAB_SIZE]
                     [-tgt_vocab_size TGT_VOCAB_SIZE]
                     [-src_words_min_frequency SRC_WORDS_MIN_FREQUENCY]
                     [-tgt_words_min_frequency TGT_WORDS_MIN_FREQUENCY]
                     [-dynamic_dict] [-share_vocab]
                     [-src_seq_length SRC_SEQ_LENGTH]
                     [-src_seq_length_trunc SRC_SEQ_LENGTH_TRUNC]
                     [-tgt_seq_length TGT_SEQ_LENGTH]
                     [-tgt_seq_length_trunc TGT_SEQ_LENGTH_TRUNC] [-lower]
                     [-shuffle SHUFFLE] [-seed SEED]
                     [-report_every REPORT_EVERY] [-sample_rate SAMPLE_RATE]
                     [-window_size WINDOW_SIZE] [-window_stride WINDOW_STRIDE]
                     [-window WINDOW]

