import numpy as np

class KNN():
  """
     K-Nearest Neighbors (KNN) Regression and Classification
  """
  def __init__(self,K=3,task='classification'):
    """
    Args:
    K : value for K, which determines how many nearest neighbors will be considered when making predictions.
    task: it can be classification or regression depend on you needs
    """
    self.K = K
    self.task = task


  def predict(self, indep_vars,output_vars, target_value):
    """
       Make Prediction for given data point
    Args:
    indep_vars: the independant variables of the dataset (X)
    output_vars: the independant variables of the dataset (y)
    target_value: the independant variables of data point that we want to make prediction from i.
    """
    def euclidean_distance(self,p_data_points, q_data_points):
      """
          Calculate the distance between the new data point and all data points in the training dataset 
      Args:
      p_data_points: it is the data point in a given dataset which consist only on the independant variables
      q_data_points: it is the data point that we want to calculate thier prediction.
      """
      euclidean_distance_ = np.sqrt(np.sum([np.square(pi-qi) for pi, qi in zip(p_data_points,q_data_points)]))
      return euclidean_distance_
    # Calculate the Euclidean Distance for each data point with the targat value
    euclidean_distances = [euclidean_distance(ind_var,target_value) for ind_var in indep_vars]
    # Sort all the distances and take the K elements
    k_nieghbors = np.argsort(euclidean_distances)[:self.K]
    if self.task == "regression":
      # Calcuate the mean of the all neighbors (y)
      prediction = np.mean(k_nieghbors)
      return prediction
    # Grap the corresponding classes of neighbore
    k_nieghbors_classes = [output_vars[k_nieghbor] for k_nieghbor in k_nieghbors]
    counts = np.bincount(k_nieghbors_classes)
    # Take the max count element as prediction class
    most_frequent_element = np.argmax(counts)
    return most_frequent_element
    
