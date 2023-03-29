from transformers import T5Tokenizer

from bert import RegressionConfig, Trainer

if __name__ == '__main__':
    # conf = RegressionConfig()
    RegressionConfig.csv_data_path = r'/home/jp/Documents/Bert/data/result-final.csv'
    RegressionConfig.drop_columns = ['uniprot_id', 'all_mutation_str']
    # RegressionConfig.max_length = 1536
    RegressionConfig.max_length = 1536
    RegressionConfig.model_name = 'Rostlab/prot_t5_xl_half_uniref50-enc'
    RegressionConfig.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
    RegressionConfig.label_set = ['E', 'N', 'H', 'T', 'F', 'I', 'R', 'S', 'K', 'Q', 'L', 'Y', 'G', 'A', 'V', 'C', 'W',
                                  'D', 'M', 'P']
    RegressionConfig.vocab_size = len(RegressionConfig.label_set)

    RegressionConfig.nr_frozen_epochs = 5
    RegressionConfig.encoder_learning_rate = 5e-06
    RegressionConfig.fine_tune_learning_rate = 3e-05
    RegressionConfig.batch_size = 4

    RegressionConfig.monitor = 'val_loss'

    RegressionConfig.checkpoint_save_path = r'/home/jp/Documents/Bert/experiments'
    RegressionConfig.max_epochs = 30
    RegressionConfig.precision = 16
    RegressionConfig.amp_level = '01'
    RegressionConfig.devices = "-1"
    RegressionConfig.accelerator = 'gpu'
    # RegressionConfig.num_processes = '30'
    # RegressionConfig.strategy = None

    t = Trainer()
    t.train()
