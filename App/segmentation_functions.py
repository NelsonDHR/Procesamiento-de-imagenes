import nibabel as nib
import numpy as np

def umbralizacion(image,tau):
  image=image
  tau=tau
  segmentation = image >=tau
  return segmentation


def isodata(image,tau):
    tol = 1
    tau = tau
    image=image

    while True:
      print(tau)
      segmentation = image >= tau
      mBG = image[np.multiply(image > 10, segmentation == 0)].mean()
      mFG = image[np.multiply(image > 10, segmentation == 1)].mean()

      tau_post = 0.5 * (mBG + mFG)

      if np.abs(tau - tau_post) < tol:
          break
      else:
          tau = tau_post
      
    return segmentation

def kmeans(image, k):
    # Create the cluster centers
    ks = np.linspace(np.amin(image), np.amax(image), k)


    for i in range(5):
        ds = [np.abs(k - image) for k in ks]
        segmentation = np.argmin(ds, axis=0)

        for j in range(k):
            ks[j] = image[segmentation == j].mean()
        print(ks)



    # Exclude background from the segmentation
    background = image < 10
    segmentation[background] = -1

    # Calculate the percentage of the non-background image corresponding to each cluster
    percentage = [np.sum(segmentation == j) / np.sum(~background) * 100 for j in range(k)]
    print("Percentage of Non-Background Image for Each Cluster:", percentage)

    return segmentation.astype(np.uint8)

def gmm(image, n_clusters):
    seg = np.zeros_like(image)
    
    # Initialize weights, means and standard deviations for each cluster
    weights = np.full(n_clusters, 1/n_clusters)
    means = np.linspace(0, 150, n_clusters)
    sds = np.full(n_clusters, 50)
    
    for iter in range(1, 5):
        # Compute likelihood of belonging to a cluster
        ps = [1/np.sqrt(2*np.pi*sds[i]**2) * np.exp(-0.5*np.power(image - means[i], 2) / sds[i]**2) for i in range(n_clusters)]
        
        # Normalize probability
        rs = [np.divide(weights[i] * ps[i], sum([weights[j] * ps[j] for j in range(n_clusters)])) for i in range(n_clusters)]
        
        # Update parameters
        weights = [rs[i].mean() for i in range(n_clusters)]
        means = [np.multiply(rs[i], image).sum() / rs[i].sum() for i in range(n_clusters)]
        sds = [np.sqrt(np.multiply(rs[i], np.power(image - means[i], 2)).sum() / rs[i].sum()) for i in range(n_clusters)]
    
    # segmentation
    seg = np.argmax(np.stack(rs, axis=-1), axis=-1)
    
    return seg

