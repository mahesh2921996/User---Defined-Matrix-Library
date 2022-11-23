# square matrix\\
def square_matrix(matrix):
    """
    This function can be used to check square matrix.

    Parameters
    ----------
    matrix : any matrix
    """
    import numpy as np
    # no of rows == no of columns :: square matrix
    if matrix.shape[0] == matrix.shape[1]:
        print('This is a square matrix.')
    else:
        print('This is not a square matrix.')


# diagonal matrix
def diagonal_matrix(matrix):
    """
        This function can be used to check diagonal matrix.

        Parameters
        ----------
        matrix : any matrix
    """
    import numpy as np
    # diagonal matrix is also square matrix.
    if matrix.shape[0] == matrix.shape[1]:

        result_list = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ## check non diagonal element must be zero
                if i != j:
                    if matrix[i, j] == 0:
                        pass
                    else:
                        result_list.append(matrix[i, j])

        # check all condition regarding to diagonal matrix, here at least 1 element is found in result list, then we say, this is not diagonal matrix
        if len(result_list) == 0:
            print('This is a diagonal matrix.')
        else:
            print('This is not a diagonal matrix.')


# scalar matrix
def scalar_matrix(matrix):
    """
        This function can be used to check scalar matrix.

        Parameters
        ----------
        matrix : any matrix
    """
    import numpy as np
    # scalar matrix is also square matrix.
    if matrix.shape[0] == matrix.shape[1]:

        result_list = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                if i != j:
                    # check non diagonal element must be 0
                    if matrix[i, j] == 0:
                        pass
                    else:
                        result_list.append(matrix[i, j])

        # check all condition regarding to scalar matrix, here at least 1 element is found in result list, then we say, this is not scalar matrix
        if len(result_list) == 0:
            print('This is a scalar matrix.')
        else:
            print('This is not a scalar matrix.')



# identity_matrix
def identity_matrix(matrix):
    """
        This function can be used to check identity matrix.

        Parameters
        ----------
        matrix : any matrix
    """
    import numpy as np
    # identity matrix is also square matrix.
    if matrix.shape[0] == matrix.shape[1]:

        result_list = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ## check diagonal element
                if i == j:
                    # check all diagonal element must be 1
                    if matrix[i, j] == 1:
                        pass
                    else:
                        result_list.append(matrix[i, j])

                else:
                    # check non diagonal element must be 0
                    if matrix[i, j] == 0:
                        pass
                    else:
                        result_list.append(matrix[i, j])

        # check all condition regarding to identity matrix, here at least 1 element is found in result list, then we say, this is not unit matrix
        if len(result_list) == 0:
            print('This is an identity matrix.')
        else:
            print('This is not an identity matrix.')


# symmetric matrix
def symmetric_matrix(matrix):
    """
        This function can be used to check symmetric matrix.

        Parameters
        ----------
        matrix : any matrix
    """
    import numpy as np
    # symmetric matrix is also square matrix.
    if matrix.shape[0] == matrix.shape[1]:

        result_list = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                if i != j:
                    # check non diagonal element are Aij=Aji
                    if matrix[i, j] == matrix[j, i]:
                        pass
                    else:
                        result_list.append(matrix[i, j])

        # check all condition regarding to symmetric matrix, here at least 1 element is found in result list, then we say, this is not symmetric matrix
        if len(result_list) == 0:
            print('This is a symmetric matrix.')
        else:
            print('This is not a symmetric matrix.')


# skew symmetric matrix
def skew_symmetric_matrix(matrix):
    """
        This function can be used to check diagonal matrix.

        Parameters
        ----------
        matrix : any matrix
    """
    import numpy as np
    # skew symmetric matrix is also square matrix.
    if matrix.shape[0] == matrix.shape[1]:

        result_list = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                ## check diagonal element
                if i == j:
                    # check all diagonal element must be 0
                    if matrix[i, j] == 0:
                        pass
                    else:
                        result_list.append(matrix[i, j])

                else:
                    # check non diagonal element are Aij == - Aji
                    if matrix[i, j] == -matrix[j, i]:
                        pass
                    else:
                        result_list.append(matrix[i, j])

        # check all condition regarding to skew symmetric matrix, here at least 1 element is found in result list, then we say, this is not symmetric matrix
        if len(result_list) == 0:
            print('This is a skew symmetric matrix.')
        else:
            print('This is not a skew symmetric matrix.')


# triangular matrix
def triangular_matrix(matrix):
    """
        This function can be used to check upper triangular or lower triangular matrix.

        Parameters
        ----------
        matrix : any matrix
    """
    import numpy as np
    # triangular matrix is also square matrix.
    if matrix.shape[0] == matrix.shape[1]:

        result_list1 = []
        result_list2 = []
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):

                if i != j:
                    # check upper triangular matrix condition
                    if i > 0 and i > j:
                        if matrix[i, j] == 0:
                            pass
                        else:
                            result_list1.append(matrix[i, j])

                    # check lower triangular matrix condition
                    if i < matrix.shape[0] and i < j:
                        if matrix[i, j] == 0:
                            pass
                        else:
                            result_list2.append(matrix[i, j])

        # check all condition regarding to triangular matrix, here at least 1 element is found in result1 and result2 list, then we say, this is not triangular matrix
        if len(result_list1) == 0:
            print('This is a upper triangular matrix.')

        elif len(result_list2) == 0:
            print('This is a lower triangular matrix.')

        else:
            print('This is not a triangular matrix.')



# orthogonal matrix
def orthogonal_matrix(matrix):
    """
        This function can be used to check orthogonal matrix.

        Parameters
        ----------
        matrix : any square matrix
    """
    import numpy as np

    # transpose matrix
    def transpose_matrix(matrix):
        result = np.zeros((matrix.shape[1], matrix.shape[0]), dtype=int)
        result = np.array(result)
        result = np.matrix(result)
        # iterate through rows
        for i in range(matrix.shape[0]):
            # iterate through columns
            for j in range(matrix.shape[1]):
                result[j, i] = matrix[i, j]

        return (result)

    matrix_t = transpose_matrix(matrix)

    # multiply original matrix by transpose matrix
    if matrix.shape[1] == matrix_t.shape[0]:

        result_matrix = np.zeros((matrix.shape[0], matrix_t.shape[1]), dtype=int)

        for i in range(matrix.shape[0]):
            for j in range(matrix_t.shape[1]):

                sum_prod_mat = 0
                for k in range(matrix_t.shape[0]):
                    sum_prod_mat = matrix[i, k] * matrix_t[k, j] + sum_prod_mat

                result_matrix[i, j] = sum_prod_mat

        # check result matrix is identity matrix or not
        if result_matrix.shape[0] == result_matrix.shape[1]:

            result_list = []
            for i in range(result_matrix.shape[0]):
                for j in range(result_matrix.shape[1]):
                    ## check diagonal element
                    if i == j:
                        # check all diagonal element must be 1
                        if result_matrix[i, j] == 1:
                            pass
                        else:
                            result_list.append(matrix[i, j])

                    else:
                        # check non diagonal element must be 0
                        if result_matrix[i, j] == 0:
                            pass
                        else:
                            result_list.append(matrix[i, j])

            # check all condition regarding to orthogonal matrix, here at least 1 element is found in result list, then we can say, this is not orthogonal matrix
            if len(result_list) == 0:
                print('This is an orthogonal matrix.')
            else:
                print('This is not an orthogonal matrix.')


# conjugate matrix
def conjugate_matrix(matrix):
    """
        This function can be used to compute the conjugate matrix.

        Parameters
        ----------
        matrix : any square matrix
    """
    import numpy as np
    # create complex matrix
    result_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=complex)
    result_matrix = np.array(result_matrix)
    result_matrix = np.matrix(result_matrix)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            result = matrix[i, j]
            # change sign of imaginary part
            conjugate = -1 * result.imag * 1j
            real = result.real
            con_element = real + conjugate

            result_matrix[i, j] = con_element
    print('the conjugate matrix is:')
    return print(f' {result_matrix}')


# Idempotent matrix
def idempotent_matrix(matrix1, stop_switch=1):
    """
        This function can be used to check idempotent matrix.

        Parameters
        ----------
        matrix : any square matrix
    """
    import numpy as np

    if matrix1.shape[1] == matrix1.shape[0]:

        # this matrix will be store A*A
        result_matrix = np.zeros((matrix1.shape[0], matrix1.shape[1]), dtype=int)

        # multiplication function
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):

                sum_prod_mat = 0
                for k in range(matrix1.shape[0]):
                    sum_prod_mat = matrix1[i, k] * matrix1[k, j] + sum_prod_mat

                result_matrix[i, j] = sum_prod_mat


        # this list store element of A*A
        result_list = []
        for m in range(0, result_matrix.shape[0]):
            for n in range(0, result_matrix.shape[1]):
                result_list.append(result_matrix[m, n])

        # this list store element of original matrix
        matrix1_list = []
        for a in range(0, matrix1.shape[0]):
            for b in range(0, matrix1.shape[1]):
                matrix1_list.append(matrix1[a, b])



        # check idempotent condition : A^2 = A*A = A or A^3 = A*A*A = A
        if matrix1_list == result_list:
            print('This matrix is an idempotent matrix.')
        else:
            if stop_switch < 10:
                stop_switch = stop_switch + 1
                idempotent_matrix(result_matrix, stop_switch)
            else:
                print('This matrix is not an idempotent matrix.')

    else:
        print("Matrix is never multiply, because 'no. of columns of 1st matrix == no. of rows of 2nd matrix' this rule not follow.")


# Periodic_matrix
def Periodic_matrix(matrix1, result_matrix1, stop_switch=1):
    """
        This function can be used to check periodic matrix.

        >>> result_matrix1 = matrix1  :: copy of original matix
        >>> Periodic_matrix(matrix1,result_matrix1)
        ... This matrix is Periodic matrix with value k

        Parameters
        ----------
        matrix1 : any square matrix
    """
    import numpy as np

    ## check multiplication rule
    if matrix1.shape[1] == matrix1.shape[0]:

        ## this matrix will be store powers of A
        result_matrix = np.zeros((matrix1.shape[0], matrix1.shape[1]), dtype=int)

        ## multiplication function
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):

                sum_prod_mat = 0
                for k in range(matrix1.shape[0]):
                    sum_prod_mat = result_matrix1[i, k] * matrix1[k, j] + sum_prod_mat

                result_matrix[i, j] = sum_prod_mat

        ## this list store element of powers of A
        result_list = []
        for m in range(0, result_matrix.shape[0]):
            for n in range(0, result_matrix.shape[1]):
                result_list.append(result_matrix[m, n])

        ## this list store element of original matrix
        matrix1_list = []
        for a in range(0, matrix1.shape[0]):
            for b in range(0, matrix1.shape[1]):
                matrix1_list.append(matrix1[a, b])


        ## check periodic matrix condition : A^(k+1) = A  :: where, k: positive integer
        if matrix1_list == result_list:
            print(f'This matrix is a periodic matrix with value k = {stop_switch}.')
        else:

            a = result_matrix
            if stop_switch < 10:
                stop_switch = stop_switch + 1
                Periodic_matrix(matrix1, a, stop_switch)
            else:

                print('This matrix is not a periodic matrix.')

    else:
        print("Matrix is never multiply, because 'no. of columns of 1st matrix == no. of rows of 2nd matrix' this rule not follow.")



# Nilpotent Matrix
def nilpotent_matrix(matrix1,result_matrix1, stop_switch=1):
    """
        This function can be used to check nilpotent matrix.

        >>> result_matrix1 = matrix1  :: copy of original matix
        >>> nilpotent_matrix(matrix1,result_matrix1)
        ... This matrix is Nilpotent matrix with value k

        Parameters
        ----------
        matrix1 : any square matrix
    """

    import numpy as np

    ## check multiplication rule
    if matrix1.shape[1] == matrix1.shape[0]:

        ## this matrix will be store powers of A
        result_matrix = np.zeros((matrix1.shape[0], matrix1.shape[1]), dtype=int)

        ## null matrix for camparing
        null_matrix = np.zeros((matrix1.shape[0], matrix1.shape[1]), dtype=int)


        ## multiplication function
        for i in range(matrix1.shape[0]):
            for j in range(matrix1.shape[1]):

                sum_prod_mat = 0
                for k in range(matrix1.shape[0]):
                    sum_prod_mat = result_matrix1[i, k] * matrix1[k, j] + sum_prod_mat

                result_matrix[i, j] = sum_prod_mat

        ## this list store element of powers of A
        result_list = []
        for m in range(0, result_matrix.shape[0]):
            for n in range(0, result_matrix.shape[1]):
                result_list.append(result_matrix[m, n])

        ## this list store element of null matrix
        null_list = []
        for a in range(0, matrix1.shape[0]):
            for b in range(0, matrix1.shape[1]):
                null_list.append(null_matrix[a, b])


        ## check nilpotent matrix condition : A^(k) = 0  :: where, k: positive integer
        if null_list == result_list:
            print(f'This matrix is a Nilpotent matrix with value k = {stop_switch + 1}.')
        else:

            a = result_matrix
            if stop_switch < 10:
                stop_switch = stop_switch + 1
                nilpotent_matrix(matrix1, a, stop_switch)
            else:

                print('This matrix is not a Nilpotent matrix.')

    else:
        print("Matrix is never multiply, because 'no. of columns of 1st matrix == no. of rows of 2nd matrix' this rule not follow.")


# singular matrix
def singular_matrix(matrix):
    """
        This function can be used to check singular matrix.

        Parameters
        ----------
        matrix : any 2x2 or 3x3 square matrix
    """
    import numpy as np
    def determinant(mat):
        size = mat.size
        if size == 4:
            determinant = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
            return determinant
        elif size == 9:
            determinant = mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) - mat[0, 1] * (
                        mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0]) + mat[0, 2] * (
                                      mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0])
            return determinant

    if determinant(matrix) == 0:
        print('This is a singular matrix.')
    else:
        print('This is not a singular matrix.')


# Adjoint Matrix
def adj_matrix(matrix):
    """
        Compute the adjoint of a 2x2 or 3x3 matrix.

        >>> matrix = [[1, 2, 3],
                      [2, 5, 7],
                      [3, 1, 2]]
        >>> adj_matrix(matrix)
        ... matrix([[  3,  -1,  -1],
                    [ 17,  -7,  -1],
                    [-13,   5,   1]])

        Parameter
        ----------
        matrix : any 2x2 or 3x3 square matrix
    """

    import numpy as np
    # transpose matrix
    def transpose_matrix(matrix):
        result = np.zeros((matrix.shape[1], matrix.shape[0]), dtype=int)
        result = np.array(result)
        result = np.matrix(result)
        # iterate through rows
        for i in range(matrix.shape[0]):
            # iterate through columns
            for j in range(matrix.shape[1]):
                result[j, i] = matrix[i, j]

        return (result)

    # adjoint for 2x2 matrix
    def adj_2_matrix(matrix):

        A = [0, 1]
        B = [0, 1]

        # result matrix for store cofactors of this matrix
        result_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                A.remove(i)
                B.remove(j)

                # calculate cofactors of matrix
                result_matrix[i, j] = (matrix[A[0], B[0]])

                A.insert(i, i)
                B.insert(j, j)

        adj_matrix = transpose_matrix(result_matrix)

        return adj_matrix

    # adjoint matrix for 3x3
    def adj_3_matrix(matrix):
        A = [0, 1, 2]
        B = [0, 1, 2]

        # result matrix for store cofactors of this matrix
        result_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                A.remove(i)
                B.remove(j)

                # calculate cofactors of matrix
                result_matrix[i, j] = ((-1) ** (i + j)) * (
                            matrix[A[0], B[0]] * matrix[A[1], B[1]] - matrix[A[0], B[1]] * matrix[A[1], B[0]])

                A.insert(i, i)
                B.insert(j, j)

        adj_matrix = transpose_matrix(result_matrix)

        return adj_matrix

    if matrix.size == 4:
        return adj_2_matrix(matrix)
    elif matrix.size == 9:
        return adj_3_matrix(matrix)


# Inverse Matrix
def inverse_matrix(matrix):
    """
        Compute the adjoint of a 2x2 or 3x3 matrix.

        >>> matrix = [[1, 2, 3],
                      [2, 5, 7],
                      [3, 1, 2]]
        >>> inverse_matrix(matrix)
        ... matrix([[-1.5,  0.5,  0.5],
                    [-8.5,  3.5,  0.5],
                    [ 6.5, -2.5, -0.5]])

        Parameter
        ----------
        matrix : any 2x2 or 3x3 square matrix
    """

    import numpy as np
    # determinant of a matrix
    def determinant(mat):
        size = mat.size
        if size == 4:
            determinant = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
            return determinant
        elif size == 9:
            determinant = mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1]) - mat[0, 1] * (
                        mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0]) + mat[0, 2] * (
                                      mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0])
            return determinant

    # adjoint of a matrix
    def adj_matrix(matrix):
        # transpose matrix
        def transpose_matrix(matrix):
            result = np.zeros((matrix.shape[1], matrix.shape[0]), dtype=int)
            result = np.array(result)
            result = np.matrix(result)
            # iterate through rows
            for i in range(matrix.shape[0]):
                # iterate through columns
                for j in range(matrix.shape[1]):
                    result[j, i] = matrix[i, j]

            return (result)

        # adjoint for 2x2 matrix
        def adj_2_matrix(matrix):

            A = [0, 1]
            B = [0, 1]

            # result matrix for store cofactors of this matrix
            result_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    A.remove(i)
                    B.remove(j)

                    # calculate cofactors of matrix
                    result_matrix[i, j] = (matrix[A[0], B[0]])

                    A.insert(i, i)
                    B.insert(j, j)

            adj_matrix = transpose_matrix(result_matrix)

            return adj_matrix

        # adjoint matrix for 3x3
        def adj_3_matrix(matrix):
            A = [0, 1, 2]
            B = [0, 1, 2]

            # result matrix for store cofactors of this matrix
            result_matrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=float)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    A.remove(i)
                    B.remove(j)

                    # calculate cofactors of matrix
                    result_matrix[i, j] = ((-1) ** (i + j)) * (
                                matrix[A[0], B[0]] * matrix[A[1], B[1]] - matrix[A[0], B[1]] * matrix[A[1], B[0]])

                    A.insert(i, i)
                    B.insert(j, j)

            adj_matrix = transpose_matrix(result_matrix)

            return adj_matrix

        if matrix.size == 4:
            return adj_2_matrix(matrix)
        elif matrix.size == 9:
            return adj_3_matrix(matrix)

    determinant = determinant(matrix)

    adj_matrix = adj_matrix(matrix)

    ## determinant of A matrix * inverse of A matrix = Unit matrix * adjoint matrix
    if determinant == 0:
        print("This matrix is a singular matrix. So, we can't calculate the inverse matrix.")
    else:
        inverse_matrix = adj_matrix / determinant

        return inverse_matrix

