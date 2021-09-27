import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg

if __name__ == '__main__':
    # Instantiate a simple SPN
    root = spn.Sum(children=[
        spn.Product(children=[
            spn.Gaussian(0, 0.5, 1.0),
            spn.Gaussian(1, -1.0, 0.5),
            spn.Gaussian(2, 0.0, 1.0)
        ]),
        spn.Product(children=[
            spn.Gaussian(0, 0.5, 1.5),
            spn.Gaussian(1, 0.0, 0.2),
            spn.Gaussian(2, 2.0, 1.0)
        ])
    ], weights=[0.8, 0.2])

    # Initialize the IDs of the SPN nodes
    spn.assign_ids(root)

    # Compute and print four moments
    print("Expectation: {}".format(spnalg.expectation(root)))
    print("Variance: {}".format(spnalg.variance(root)))
    print("Skewness: {}".format(spnalg.skewness(root)))
    print("Kurtosis: {}".format(spnalg.kurtosis(root)))
