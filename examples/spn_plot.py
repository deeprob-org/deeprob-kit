import deeprob.spn.structure as spn
import deeprob.spn.algorithms as spnalg

if __name__ == '__main__':
    # Instantiate a dummy SPN
    b1 = spn.Bernoulli(1, p=0.1)
    b2 = spn.Bernoulli(0, p=0.5)
    p0 = spn.Product(children=[spn.Bernoulli(0, p=0.9), b1])
    p1 = spn.Product(children=[b2, spn.Bernoulli(1, p=0.5)])
    s0 = spn.Sum(children=[
        spn.Bernoulli(0, p=0.1), spn.Bernoulli(0, p=0.4)
    ], weights=[0.3, 0.7])
    root = spn.Sum(children=[
        p0,
        spn.Product(children=[s0, spn.Product(children=[b1])]),
        spn.Sum(children=[p0, p1], weights=[0.6, 0.4])
    ], weights=[0.3, 0.5, 0.2])

    # Initialize the IDs of the SPN nodes
    spn.assign_ids(root)

    # Plot the SPN
    spn_filename = 'spn-dummy.svg'
    print("Plotting the dummy SPN to {} ...".format(spn_filename))
    spn.plot_spn(root, spn_filename)

    # Prune and plot the SPN structure
    pruned_spn = spnalg.prune(root)
    pruned_spn_filename = 'spn-pruned.svg'
    print("Plotting the pruned SPN to {} ...".format(pruned_spn_filename))
    spn.plot_spn(pruned_spn, pruned_spn_filename)

    # Marginalize and plot the SPN structure w.r.t. the random variable 0
    marginalized_spn = spnalg.marginalize(root, [0])
    marginalized_spn_filename = 'spn-marginalized.svg'
    print("Plotting the marginalized SPN w.r.t. RV #0 to {} ...".format(marginalized_spn_filename))
    spn.plot_spn(marginalized_spn, marginalized_spn_filename)
