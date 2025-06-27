import numpy as np

class TruncatedSVDFromScratch():
    def __init__(self, n_components):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.V = None
        self.explained_variance_ratio_ = None
        
        
    def fit(self, A):
        # X là ma trận đầu vào
        A = A.astype(float)
        m, n = A.shape
        # Bước 1: Xây dựng ma trận AtA
        At_A = A.T @ A
        
        # Bước 2: Phân tích trị riêng và vector riêng của ma trận ATA
        eigenvalues, eigenvectors = np.linalg.eigh(At_A)
        
        # Bước 3: Sắp xếp eigenvalues theo thứ tự giảm dần
        sorted_idx = np.argsort(-np.abs(eigenvalues)) #np.argsort: lấy chỉ số sắp xếp
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Bước 4: Lấy top k trị riêng lớn nhất
        singular_values = np.sqrt(eigenvalues[:self.n_components])
        V = eigenvectors[:, :self.n_components]
        
        # Bước 5: Tính U_k = AV/lambda
        U = np.zeros((m, self.n_components))
        for i in range(self.n_components):
            if singular_values[i] > 0:
                U[:, i] = (A@V[:, i])/singular_values[i]
            else:
                U[:, i] = A @ V[:, i]

        self.U = U
        self.S = singular_values
        self.V = V
        self.explained_variance_ratio_ = (singular_values ** 2)/ np.sum(singular_values ** 2)
        return self
        
    def transform(self, A):
        # Giảm chiều ma trận X
        if (self.U is None) or (self.S is None) or (self.V is None):
            print("Fit again.")
        else:
            #self.svd_matrix = np.dot(A, self.V_t.T)
            return np.dot(A, self.V)
    
    def fit_transform(self, A):
        return self.fit(A).transform(A)