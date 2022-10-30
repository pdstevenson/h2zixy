def h2zixy(hamiltonian):
    """Decompose square real symmetric matrix into Pauli spin matrices

    argument:
    hamiltonian -- a square numpy real symmetric numpy array

    returns:
    a string consisting of terms each of which has a numerical coefficient
    multiplying a Kronecker (tensor) product of Pauli spin matrices
    """

    import itertools
    import numpy as np
    
    # coefficients smaller than eps are taken to be zero
    eps = 1.e-5
    
    dim = len(hamiltonian)
    
    # Step 1:expand Hamiltonian to have leading dimension = power of 2 and pad
    # with zeros if necessary
    
    NextPowTwo = int(2**np.ceil(np.log(dim)/np.log(2)))
    if NextPowTwo != dim:
        diff = NextPowTwo - dim
        hamiltonian = np.hstack((hamiltonian,np.zeros((dim, diff))))
        dim = NextPowTwo
        hamiltonian = np.vstack((hamiltonian,np.zeros((diff,dim))))
        
    # Step 2: Generate all tensor products of the appropriate length with 
    # all combinations of I,X,Y,Z, excluding those with an odd number of Y
    # matrices

    # Pauli is a dictionary with the four basis 2x2 Pauli matrices
    Pauli = {'I' : np.array([[1,0],[0,1]]),
             'X': np.array([[0,1],[1,0]]),
             'Y': np.array([[0,-1j],[1j,0]]),
             'Z': np.array([[1,0],[0,-1]])}
    
    NumTensorRepetitions = int(np.log(dim)/np.log(2))
    NumTotalTensors = 4**NumTensorRepetitions
    PauliKeyList = []
    KeysToDelete = []
    PauliDict = {}

    def PauliDictValues(l):
        yield from itertools.product(*([l] * NumTensorRepetitions))
    
    #Generate list of tensor products with all combinations of Pauli
    # matrices i.e. 'III', 'IIX', 'IIY', etc.
    for x in PauliDictValues('IXYZ'):
        PauliKeyList.append(''.join(x))

    for y in PauliKeyList:
        PauliDict[y] = 0

    for key in PauliDict:
        TempList = []
        PauliTensors = []
        NumYs= key.count('Y')
        TempKey = str(key)

        if (NumYs % 2) == 0:
            for string in TempKey:
                TempList.append(string)

            for SpinMatrix in TempList:
                PauliTensors.append(Pauli[SpinMatrix])
            PauliDict[key] = PauliTensors

            CurrentMatrix = PauliDict[key].copy()

            # Compute Tensor Product between I, X, Y, Z matrices
            for k in range(1, NumTensorRepetitions):
                TemporaryDict = np.kron(CurrentMatrix[k-1], CurrentMatrix[k])
                CurrentMatrix[k] = TemporaryDict

            PauliDict[key] = CurrentMatrix[-1]

        else:
            KeysToDelete.append(key)
    
    for val in KeysToDelete:
        PauliDict.pop(val)

    # Step 3:  Loop through all the elements of the Hamiltonian matrix
    # and identify which pauli matrix combinations contribute; 
    # Generate a matrix of simultaneous equations that need to be solved.
    # NB upper triangle of hamiltonian array is used

    VecHamElements = np.zeros(int((dim**2+dim)/2))
    h = 0
    for i in range(0,dim):
        for j in range(i,dim):
            arr = []
            VecHamElements[h] = hamiltonian[i,j]
            for key in PauliDict:
                TempVar = PauliDict[key]
                arr.append(TempVar[i,j].real)

            if i == 0 and j == 0:
                FinalMat = np.array(arr.copy())

            else:
                FinalMat = np.vstack((FinalMat, arr))

            h += 1

    # Step 4: Use numpy.linalg.solve to solve the simultaneous equations
    # and return the coefficients of the Pauli tensor products.

    x = np.linalg.solve(FinalMat,VecHamElements)
    a = []
    var_list = list(PauliDict.keys())

    for i in range(len(PauliDict)):
        b = x[i]
        if  abs(b)>eps:
            a.append(str(b)+'*'+str(var_list[i])+'\n')
    
    # Output the final Pauli Decomposition of the Hamiltonian
    DecomposedHam = ''.join(a)
    return DecomposedHam


if __name__ == '__main__':
    import numpy as np

    # for a sample calculation, take the Hamiltonian from the paper by
    # Dumitrescu et al. (Phys. Rev. Lett. 120, 210501 (2018)) 

    N = 8
    hw = 7.0
    v0 = -5.68658111
    
    ham = np.zeros((N,N))
    ham[0,0] = v0
    for n in range (0,N):
        for na in range(0,N):
            if(n==na):
                ham[n,na] += hw/2.0*(2*n+1.5)
            if(n==na+1):
                ham[n,na] -= hw/2.0*np.sqrt(n*(n+0.5))
            if(n==na-1):
                ham[n,na] -= hw/2.0*np.sqrt((n+1.0)*(n+1.5))

    out= h2zixy(ham)
    print(out)
