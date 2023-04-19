import chemprop


arguments = [
                '--data_path', 'tmp/MPNN/train.csv',
                '--separate_test_path', 'tmp/MPNN/test.csv',
                '--separate_val_path', 'tmp/MPNN/test.csv',
                '--dataset_type', 'regression',
                '--save_dir', 'tmp/MPNN/test_checkpoints_reg',
                '--epochs', '30',
                '--num_folds', '1',
                '--ffn_num_layers', '3',
            ]

args2 = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args2, train_func=chemprop.train.run_training)
