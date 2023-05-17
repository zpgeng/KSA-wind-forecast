class Hyperpara:
    
    def __init__(self):
        
        # Set the optimal parameter values:
        # m, n_h, lambda, delta, phi, a_w, a_u, phi_w, phi_u
        self.parameter = [1, 2500, 0.150, 0.90, 1.00, 0.05, 0.01, 0.10, 0.01]

        # Set additional parameters for NNF implementation: 
        # self.parameter.extend([5, 4, 0.05])

        print("The optimal values of ESN parameters:\n")
        print("                         input lag,      m =", self.parameter[0])
        print("        number of reservoir states,    n_h =", self.parameter[1])
        print("                     ridge penalty, lambda =", self.parameter[2])
        print("    scaling matrix parameter for W,  delta =", self.parameter[3])
        print("                      leaking rate,    phi =", self.parameter[4])
        print("         magnitude of entries in W,    a_w =", self.parameter[5])
        print("         magnitude of entries in U,    a_u =", self.parameter[6])
        print("                     sparsity of W,   pi_w =", self.parameter[7])
        print("                     sparsity of U,   pi_u =", self.parameter[8])
       # print("number of nearest neighbour filters,   n_f =", self.parameter[9])
       # print("         number of nearest neighbours,   k =", self.parameter[10])
       # print("         magnitude of entries in V,    a_v =", self.parameter[11])