from CMK import default_args, main

data_names = ['BBC2', 'bbcsport_2view', 'CiteSeer', 'Cora', 'Movies']
normalizes = [True, False]
dims = [4, 8, 16, 32, 64, 128, 256, 512]
learning_rates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
epochs_set = [150, 300, 450]

for id in range(len(data_names)):
    for normalize in normalizes:
        for dim in dims:
            for lr in learning_rates:
                for epochs in epochs_set:

                    data_name = data_names[id]
                    print('# {}, norm_{}, dim_{}, lr_{}, epochs_{}'.format(data_name, normalize, dim, lr, epochs))

                    # Gaussian
                    print(" - Gaussian")
                    args = default_args(data_name, normalize, dim, lr, epochs)
                    args.kernel_options['type'] = 'Gaussian'
                    args.kernel_options['t'] = 1.0
                    main(args)

                    # Linear
                    print(" - Linear")
                    args = default_args(data_name, normalize, dim, lr, epochs)
                    args.kernel_options['type'] = 'Linear'
                    main(args)

                    # Polynomial
                    print(" - Polynomial")
                    args = default_args(data_name, normalize, dim, lr, epochs)
                    args.kernel_options['type'] = 'Polynomial'
                    args.kernel_options['a'] = 1.0
                    args.kernel_options['b'] = 1.0
                    args.kernel_options['d'] = 2.0
                    main(args)

                    # Sigmoid
                    print(" - Sigmoid")
                    args = default_args(data_name, normalize, dim, lr, epochs)
                    args.kernel_options['type'] = 'Sigmoid'
                    args.kernel_options['d'] = 2.0
                    args.kernel_options['c'] = 0.0
                    main(args)

                    # Cauchy
                    print(" - Cauchy")
                    args = default_args(data_name, normalize, dim, lr, epochs)
                    args.kernel_options['type'] = 'Cauchy'
                    args.kernel_options['sigma'] = 1.0
                    main(args)
