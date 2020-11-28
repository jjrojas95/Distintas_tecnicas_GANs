# Distintas_tecnicas_GANs
Este es un repositorio con distintas técnicas para el entrenamiento y construcción de Generative Adversarial Networks o por sus siglas en inglés GAN's relacionada con la creación de imágenes.

# ¿Que encontrará?
En el directorio *notebooks* encontrará una serie de carpetas con distintos capítulos en los que se busca replicar los resultados y explicar los conceptos de distintos artículos referentes a las **redes generativas antagónicas** o por como se le conocen en inglés *Generative Adversarial Networks* (GAN). Todo esto plasmado en distintos notebooks.

## Capítulo 1
Este capítulo se basa sobre todo en el artículo inicial de este marco de referencia llamado [*Generative Adversarial Nets*](https://arxiv.org/pdf/1406.2661.pdf) escrito por los investigadores de la Universidad de Montreal Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley,Sherjil Ozair, Aaron Courville, Yoshua Bengio. También se apoya en otros artículos que brindan herramientas para poder replicar los resultados del primer artículo. A continuación algunos de los resultados con el conjunto de datos MNIST. El número enmarcado en verde corresponde al ejemplo del conjunto de entrenamiento que mas se acerca a la imagen a la izquierda de éste al final del entrenamiento. Recuerde es el que mas se acerca, mas no significa que son iguales.

![entrenamiento MNIST](./notebooks/GAN_capitulo_1/Entrenamiento_GAN_Mnist.gif)

### Docker
Si quiere ejecutar el entorno con los distintos requerimiento de librerías y con jupyter notebook puede construir la imagen y crear el container de la carpeta asignada a cada capítulo de las GAN's. Para el capítulo 1 puede construir la imagen de la siguiente manera:

`docker build -t gan_cap_1 -f notebooks/GAN_capitulo_1/Dockerfile notebooks/GAN_capitulo_1`

Nuestra imagen recibirá el nombre de **gan_cap_1**. A continuación creará el container con el siguiente comando:

`docker run -p 8888:8888 -v $(pwd)/notebooks/GAN_capitulo_1/:/home/GAN_cap1/ gan_cap_1`

En su shell le aparecerá una serie de instruciones lo que debe haecr es acceder a la ruta mostrada por el shell que tiene la siguiente forma: `http://127.0.0.1:8888/?token=${SU_TOKEN}`

## Capítulo 2
En construcción

# Requeimientos
* Python 3.5 - 3.8
* Tensorflow 2.0 o mayor
* [Numpy](https://numpy.org/)
* [matplotlib](https://matplotlib.org/)
* [glob](https://docs.python.org/3/library/glob.html)
