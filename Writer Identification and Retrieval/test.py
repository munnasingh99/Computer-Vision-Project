import numpy as np
import cv2
from tqdm import tqdm
import gzip
import _pickle as cPickle

codebook_model = MiniBatchKMeans(n_clusters=k)
codebook_model.fit(selected_descriptors)

# Get the codebook (dictionary)
codebook = codebook_model.cluster_centers_


# Compute the full association matrix
association_matrix = []
bf = cv2.BFMatcher()
for descriptor in descriptors:
    matches = bf.knnMatch(descriptor, codebook, k=1)
    association_vector = [0] * k
    for match in matches:
        cluster_index = match[0].trainIdx
        association_vector[cluster_index] = 1
    association_matrix.append(association_vector)
    
    

    def assignments(descriptors, clusters):
        """ 
        compute assignment matrix
        parameters:
            descriptors: TxD descriptor matrix
            clusters: KxD cluster matrix
        returns: TxK assignment matrix
        """
        # compute nearest neighbors
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(descriptors,clusters,k=1)

        assignment = np.zeros((len(descriptors),len(clusters)))
        for i in range(len(matches)):
            assignment[i][matches[i][0].trainIdx] = 1
            
        return assignment

    def vlad(files, mus, powernorm, gmp=False, gamma=1000):
        """
        compute VLAD encoding for each files
        parameters: 
            files: list of N files containing each T local descriptors of dimension
            D
            mus: KxD matrix of cluster centers
            gmp: if set to True use generalized max pooling instead of sum pooling
        returns: NxK*D matrix of encodings
        """
        K = mus.shape[0]
        encodings = []

        for f in tqdm(files):
            with gzip.open(f, 'rb') as ff:
                desc = cPickle.load(ff, encoding='latin1')
            a = assignments(desc, mus)
            
            T,D = desc.shape
            f_enc = np.zeros( (D*K), dtype=np.float32)
            for k in range(mus.shape[0]):
                # it's faster to select only those descriptors that have
                # this cluster as nearest neighbor and then compute the 
                # difference to the cluster center than computing the differences
                # first and then select
                # TODO
                cluster_descs = desc[a[:, k] == 1]

                # Compute VLAD residuals
                residuals = cluster_descs - mus[k]

                # Sum pooling or generalized max pooling
                if not gmp:
                    f_enc[k*D:(k+1)*D] = np.sum(residuals, axis=0)
                else:
                    # TODO: Implement generalized max pooling
                    # You may need to refer to the original VLAD paper for this
                    pass

            # c) Power normalization
            if powernorm:
                f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))

            # L2 normalization
            f_enc /= np.linalg.norm(f_enc)

            encodings.append(f_enc)

        return encodings

    # Usage example
    files = ['file1', 'file2', 'file3']
    mus = np.random.rand(10, 128)
    powernorm = True
    gmp = False
    gamma = 1000

    encodings = vlad(files, mus, powernorm, gmp, gamma)


