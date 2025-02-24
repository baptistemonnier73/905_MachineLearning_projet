# Ébauche d'IA capable de reconnaître des célébrités

## But

Reconnaître des personnes célèbres sur des photos. Notre dataset est issue de huggingface: tonyassi/celebrity-1000



## Exécution

Pour l'exécution de cette ébauche, nous recommandons Kaggle avec les paramètres suivants : 
 - Accès à internet activé
 - 2 GPUs T4

Commencez par exécuter les commandes suivantes :
````
!pip install transformers torch torchvision pillow
!pip install -q accelerate datasets peft bitsandbytes tensorboard
!pip install -q flash-attn --no-build-isolation
````

Copiez-collez ensuite le fichier [init.py](./src/init.py) dans une cellule kaggle et exécutez-la.
Ce fichier contient un script permettant l'initialisation du projet

Pour continuer, copiez-collez le fichier [train.py](./src/train.py) dans une cellule kaggle et exécutez-la.
Ce fichier contient la fonction d'entrainement ainsi que son appel. Vous pouvez modifier ces trois variables présentes vers la fin :
````python
name = "Aaron Taylor-Johnson" # nom de la célébrité avec laquelle entraîner le modèle
num_train_epochs = 2 # nombre d'epoch
train_dataset_size = 5 # taille du dataset (prendra ici les 5 premiers éléments du dataset présent sur huggingface
````

Enfin, exécutez le contenu du fichier [ui.py](./src/ui.py) qui est une fonction permettant de tester l'entrainement précédemment effectué et son appel.
Vous pouvez modifier cette variable présente vers la fin :

````python
name = "Aaron Taylor-Johnson" # appel avec cette célébrité : prends une image de cette célébrité pour faire un test avec et devrait donner en sortie le même nom que celui passé en paramètres avec un modèle suffisament entraîné
````