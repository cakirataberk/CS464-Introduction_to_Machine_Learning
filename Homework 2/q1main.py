# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:20:52 2021

@author: User
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#read data as matrix form from csv 
images = pd.read_csv("images.csv") 
images = images.values
meaned_images = images - np.mean(images,axis = 0) 


def PCA (images, num_principle_components):
    pve = []
    #Mean center the data
    meaned_images = images - np.mean(images,axis = 0) 
    #Compute covariance matrix
    covar_matrix = np.cov(meaned_images,rowvar  = False) #set rowvar to False to get covariance matrix in required dims
    #Calculate eigenvalues and eigenvectors of covcariance matrix
    eigen_values, eigen_vector = np.linalg.eigh(covar_matrix)
    #sorting eigen vectors and eigen values 
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vector[:,sorted_index]
    sorted_eigen_values = eigen_values[sorted_index]
    #select first n principle components 
    principle_components = sorted_eigen_vectors[:,0:num_principle_components]
    eigen_values = sorted_eigen_values[:num_principle_components]
    #calculate PVE
    for i in eigen_values:
        pve.append(i/sum(sorted_eigen_values)*100)
    return principle_components,pve

def reshape_components(principle_components,num_principle_components):
    reshaped_principle_components = []
    for i in range (num_principle_components):
        pca = np.reshape(principle_components[:,i],(48,48))
        reshaped_principle_components.append(pca)
    return reshaped_principle_components
    
def show_pca(num_principle_components):
    plt.figure(1) 
    for i in range(num_principle_components):
        plt.subplot(2, 5, i+1)
        plt.imshow(reshaped_components[i],cmap="gray")

def print_pve(pve):
    j = 1
    for i in pve :
        print("PVE of principle component "+str(j)+" is % "+str(i))
        j+=1
        
def plot_pve():
    num_principle_components = [1, 10, 50, 100, 500]
    pves = []
    for k in num_principle_components:
        principle_components,pve = PCA (images, k)
        pves.append(sum(pve))
        print("for k equals "+str(k)+" pve is % "+ str(sum(pve)))
    plt.figure(2) 
    plt.scatter(num_principle_components,pves,color = "blue")
    plt.title("k vs PVE")
    plt.xlabel("k")
    plt.ylabel("PVE")
    plt.show()

def reconstruct_image():
    num_principle_components = [1, 10, 50, 100, 500]
    i=1
    plt.figure(3) 
    for k in num_principle_components:  
        principle_components,pve = PCA (images,k)
        eigen_vec = principle_components[:,:k]
        eigen_k_transpose = eigen_vec.transpose()
        project = np.dot(meaned_images[0], eigen_vec)
        recon = np.dot(project,eigen_k_transpose)
        recon = np.add(recon, np.mean(meaned_images, axis = 0))
        plt.subplot(1, 5, i)
        plt.imshow(np.reshape(recon,(48,48)),cmap="gray")
        i+=1
        
principle_components,pve = PCA (images,10)
reshaped_components = reshape_components(principle_components,10)
print_pve(pve)
show_pca(10)
plot_pve()
reconstruct_image()





