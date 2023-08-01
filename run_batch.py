from CMK_batch import default_args, main

data_names = ['AwA_sel_fea', 'CCV_fea', 'NUS-WIDE-OBJECT_fea', 'YoutubeFace_3view_fea', 'Winnipeg1_fea']
normalizes = [True]
dims = [128]
batch_sizes = [1024]
learning_rates = [1]
epochs_set = [90, 150, 300]

for epochs in epochs_set:
    for normalize in normalizes:
        for dim in dims:
            for batch_size in batch_sizes:
                for learning_rate in learning_rates:
                    for id in range(len(data_names)):


                        data_name = data_names[id]
                        print('# {}, norm_{}, dim_{}, batch_size_{}, lr_{}, epochs_{}'.format(data_name, normalize, dim, batch_size, learning_rate, epochs))

                        # Gaussian
                        print(" - Gaussian")
                        args = default_args(data_name, normalize, dim, batch_size, learning_rate, epochs)
                        args.kernel_options['type'] = 'Gaussian'
                        args.kernel_options['t'] = 1.0
                        main(args)

                        # Linear
                        print(" - Linear")
                        args = default_args(data_name, normalize, dim, batch_size, learning_rate, epochs)
                        args.kernel_options['type'] = 'Linear'
                        main(args)
                        
                        # Polynomial
                        print(" - Polynomial")
                        args = default_args(data_name, normalize, dim, batch_size, learning_rate, epochs)
                        args.kernel_options['type'] = 'Polynomial'
                        args.kernel_options['a'] = 1.0
                        args.kernel_options['b'] = 1.0
                        args.kernel_options['d'] = 2.0
                        main(args)

                        # Sigmoid
                        print(" - Sigmoid")
                        args = default_args(data_name, normalize, dim, batch_size, learning_rate, epochs)
                        args.kernel_options['type'] = 'Sigmoid'
                        args.kernel_options['d'] = 2.0
                        args.kernel_options['c'] = 0.0
                        main(args)

                        # Cauchy
                        print(" - Cauchy")
                        args = default_args(data_name, normalize, dim, batch_size, learning_rate, epochs)
                        args.kernel_options['type'] = 'Cauchy'
                        args.kernel_options['sigma'] = 1.0
                        main(args)
