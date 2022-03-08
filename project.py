import os
import argparse
import json
import numpy as np
import torch
from s2v_code.Screen2Vec import Screen2Vec
from sentence_transformers import SentenceTransformer
from s2v_code.prediction import TracePredictor
from s2v_code.autoencoder import ScreenLayout, LayoutAutoEncoder
from s2v_code.UI_embedding.UI2Vec import HiddenLabelPredictorModel, UI2Vec
from s2v_code.dataset.playstore_scraper import get_app_description
from s2v_code.dataset.rico_utils import get_all_labeled_uis_from_rico_screen, ScreenInfo
from s2v_code.dataset.rico_dao import load_rico_screen_dict

import pandas
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from datetime import datetime
from itertools import compress
import random

# datetime object containing current date and time
now = datetime.now()

# path to important folders (may need to be changed)
screenvec = os.path.join(os.getcwd(), "s2v_code")
rico_dataset = os.path.join(os.path.dirname(os.getcwd()), \
"RICO_CLAY_data", "unique_uis.tar", "unique_uis", "combined")

pretrained_model = os.path.join(screenvec, "s2v_pre-trained_model")
saved_tensors = os.path.join(os.getcwd(), "rico_tensors")

# path of pre-trained model
gui_element_path = os.path.join(pretrained_model, "UI2Vec_model.ep120")
screen2vec_path = os.path.join(pretrained_model,"Screen2Vec_model_v4.ep120")
layout_encoder = os.path.join(pretrained_model,"layout_encoder.ep800")

# path of the saved plots and figures
saved_plt_path = os.path.join(os.getcwd(), "results")

# path for saving the results
records = "records.txt"
records_path = os.path.join(saved_plt_path, records)



# Generates the vector embeddings for an input screen
# net version decides 
def get_embedding_project(screen, ui_model, screen_model, layout_model, num_predictors, net_version):

    # uses the screen and obtains the data if similar to rico image?
    with open(screen) as f:
        rico_screen = load_rico_screen_dict(json.load(f))
    labeled_text = get_all_labeled_uis_from_rico_screen(rico_screen)

    bert = SentenceTransformer('bert-base-nli-mean-tokens')
    bert_size = 768

    ### 1. GUI COMPONENT EMBEDDER
    ## 1. UI MODEL - GUI element embedding model.
    # initializes the model
    loaded_ui_model = HiddenLabelPredictorModel(bert, bert_size, 16)
    loaded_ui_model.load_state_dict(torch.load(ui_model), strict=False)

    # obtains class and text of components from the layout of screenshot, and creates variable [text, class]
    ui_class = torch.tensor([UI[1] for UI in labeled_text])
    ui_text = [UI[0] for UI in labeled_text]

    # obtains ambedding for all components in the UI
    UI_embeddings = loaded_ui_model.model([ui_text, ui_class])

    #avg_embedding = UI_embeddings.sum(dim=0)/len(labeled_text)

    ### 2.1 SCREEN HERARCHY - APP DESCRIPTION
    # gets description from app's google play metadata, if not, it leaves blank
    try:
        package_name = rico_screen.activity_name.split("/")[0]
        descr = get_app_description(package_name)
    except Exception as e:
        descr = ''
        print(str(e))

    # encodes description using BERT model - APP DESCRIPTION IN IMAGE
    descr_emb = torch.as_tensor(bert.encode([descr]), dtype=torch.float)
    
    ### 2.2 LAYOUT AUTOENCODER
    ## 3. LAYOUT MODEL
    # initializes and loads layout autoencoder, and encoding
    layout_autoencoder = LayoutAutoEncoder()
    layout_autoencoder.load_state_dict(torch.load(layout_model))
    layout_embedder = layout_autoencoder.enc

    # gets pixels from screen and encoded layout from screen sample
    screen_to_add = ScreenLayout(screen)
    screen_pixels = screen_to_add.pixels.flatten()
    encoded_layout = layout_embedder(torch.as_tensor(screen_pixels, dtype=torch.float).unsqueeze(0)).squeeze(0)


    if net_version in [0,2,6]:
        adus = 0
    else:
        # case where coordinates are part of UI rnn
        adus = 4
    if net_version in [0,1,6]:
        adss = 0
    else:
        # case where screen layout vec is used
        adss = 64
    if net_version in [0,1,2,3]:
        desc_size = 768
    else:
        # no description in training case
        desc_size = 0

    ### 2.3 
    # initializes and loads screen embedder from parameter
    # 2... SCREEN MODEL
    screen_embedder = Screen2Vec(bert_size, adus, adss, net_version)
    loaded_screen_model = TracePredictor(screen_embedder, net_version)
    loaded_screen_model.load_state_dict(torch.load(screen_model))


    if net_version in [1,3,4,5]:
        coords = torch.FloatTensor([labeled_text[x][2] for x in range(len(UI_embeddings))])
        UI_embeddings = torch.cat((UI_embeddings.cpu(),coords),dim=1)
    if net_version in [0,1,6]:
        screen_layout = None
    else: screen_layout = encoded_layout.unsqueeze(0).unsqueeze(0)

    # computes de Screen2Vec embedding using the GUI component embeddings, description embedding, and layout encoder.
    screen_emb = screen_embedder(UI_embeddings.unsqueeze(1).unsqueeze(0), descr_emb.unsqueeze(0), None, screen_layout, False)

    if descr_emb.size()[0] == 1:
        descr_emb = descr_emb.squeeze(0)
    #baseline_emb = torch.cat((avg_embedding, descr_emb), dim=0)
    
    #print(encoded_layout.size())
    #print((screen_emb[0][0]).size())
    #return encoded_layout, screen_emb[0][0]
    return screen_emb[0][0]



def save_embeddings(a: int, b: int) -> None:
    """Computes the embedding vectors of the screenshot layouts named from a to b, 
       and saves them as ".pt" files at 'saved_tensors' folder."""
    
    for i in range(a, b):
        print(i)

        try:
            # gets and computes the layout embedding vector
            layout = os.path.join(rico_dataset, f"{i}.json")
            embedding = get_embedding_project(layout, gui_element_path, screen2vec_path, layout_encoder, 4, 4)

            # saves the tensor at "save_tensor"
            save_tensor = os.path.join(saved_tensors, f"{i}.pt")
            torch.save(embedding, save_tensor)
        
        except FileNotFoundError:
            pass


def get_embeddings(a: int, b: int) -> [torch.Tensor]:
    """Gets the embedding vectors of the app's screenshots "a".jpg to "b".jpg
       saved in the files "a".pt to "b".pt."""
    
    embeddings = []
    
    # loads the tensors at "saved_tensors" from a to b
    for i in range(a, b):
        try:
            iembed = os.path.join(saved_tensors, f"{i}.pt")
            embeddings.append(torch.load(iembed))
        except FileNotFoundError:
            pass
    return embeddings

def get_random_examples(labels, j, k):
    """Gets k random screenshots that belongs to the 'i' cluster."""

    #indexes = list(map(int, labels==True))
    indexes = [i for i, x in enumerate(list(labels)) if x==j]
    a = random.choices(indexes, k=k)
    return a

def compute_kmeans_sse(embeddings: np.ndarray, n:int) -> list[int]:
    '''Computes SSE (sum squared error) vs number of clusters using KMeans from 1 to n and find
    the "elbow point", the point after which the SSE or inertia starts decreasing in linear fashion'''
    
    clusters = list(range(1, n))
    sse = []
    for k in clusters:
        kmeans = KMeans(n_clusters = k).fit(embeddings)
        sse.append(kmeans.inertia_)
    
    plt.plot(clusters, sse, marker="o")
    plt.title("K-Means SSE")
    plt.grid(True)
    plt.savefig(os.path.join(saved_plt_path, "SSE_n°_of_clusters"))
    plt.show()
    return sse


def main():
    record_res = open(records_path, 'a')
    record_res.write("Records :"+ now.strftime("%H:%M %Y/%m/%d")+"\n")
    #save_embeddings(2000, 3000)
    #616
    
    a,b=0,3000
    embeddings = torch.stack(get_embeddings(a,b), dim=0)
    record_res.write(f"Results from RICO screenshots {a}-{b}: {b-a}"+ "\n")

    numpy_embeddings = embeddings.detach().numpy()

    # compute SSE (sum squared error) vs number of clusters and find the "elbow point", the point after
    # which the SSE or inertia starts decreasing in linear fashion
    sse = compute_kmeans_sse(numpy_embeddings, 30)

    # apply k means
    #kmeans_n_clusters = 10
    #kmeans = KMeans(n_clusters = kmeans_n_clusters, n_init = 20, random_state = 10)
    #kmeans.fit(numpy_embeddings)

    # apply PCA to reduce the dimensionality of the embeddings
    reduced_embeddings = PCA(n_components = 2, random_state=2).fit_transform(numpy_embeddings)
    
    plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], c=kmeans.labels_)
    plt.title("K-Means")
    plt.savefig(os.path.join(saved_plt_path, "Kmeans"))
    plt.show()

    eps = 0.5
    min_samples = 5

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced_embeddings)
    plt.scatter(reduced_embeddings[:,0], reduced_embeddings[:,1], c=dbscan.labels_)
    plt.title(f"DBSCAN: eps={eps}, min_samples={min_samples}")
    plt.savefig(os.path.join(saved_plt_path, "DBSCAN"))
    plt.show()

    dbscan_2 = DBSCAN(eps=eps, min_samples=min_samples).fit(numpy_embeddings)
    labels = dbscan_2.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    
    record_res.write("DBSCAN\n")
    str_clusters = "Estimated number of clusters: %d" % n_clusters_
    str_noise = "Estimated number of noise points: %d" % n_noise_

    record_res.write(str_clusters + "\n")
    record_res.write(str_noise + "\n")
    
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    # Agglomerative Clustering
    # for clusters from 3 to 8, and the 4 linkage criteria
    n_clusters = range(3,8)
    linkage = ["ward", "average", "complete", "single"]


    save_labels = []
    for k in n_clusters:
        for criteria in linkage:
            clust = AgglomerativeClustering(n_clusters=k, linkage=criteria).fit(reduced_embeddings)
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clust.labels_)
            plt.title(f"Hierarchical: {criteria}, n_clusters={k}")
            plt.savefig(os.path.join(saved_plt_path, f"AgglomerativeClustering~{k}~{criteria}"))
            plt.show()
            
            if (k == 5) and (criteria == "average"):
                save_labels = clust.labels_

    print(save_labels)
    print("1: ", save_labels==1)

    print("Finishing Program")
    record_res.write("\n\n")
    record_res.close()

def main2():
    record_res = open(records_path, 'a')
    record_res.write("Records :"+ now.strftime("%H:%M %Y/%m/%d")+"\n")
    
    a,b=0,3000
    embeddings = torch.stack(get_embeddings(a,b), dim=0)
    record_res.write(f"Results from RICO screenshots {a}-{b}: {b-a}"+ "\n")

    numpy_embeddings = embeddings.detach().numpy()

    reduced_embeddings = PCA(n_components = 2, random_state=2).fit_transform(numpy_embeddings)
    clust = AgglomerativeClustering(n_clusters=3, linkage="average").fit(reduced_embeddings)
    labels = clust.labels_
    get_random_examples(labels, 2, 5)
    #print(n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0))



if __name__ == "__main__":
    main2()

#tensor1 = os.path.join(saved_tensors, "3.pt")
#torch.save(embeddings, tensor1)

#tensor1_load = torch.load(tensor1)
#print(tensor1_load)
#print(tensor1_load.size())
#print(embeddings)