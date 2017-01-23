# squeeznet-nano
SEM/TEM image shape classification using squeeznet

Courtsey: https://github.com/rcmalli/keras-squeezenet

In this project I used transfer learning to teach a *modified* squeeznet  to classify scanning electron microscope (SEM) and transmission electron microscope (TEM) images of nanomaterials. There are three main predicting classes nanotubes, nanoparticles and random images(None).

# Modified Squeeznets

The last softmax layer in the original squeeznet was removed and added additonal two dense layers and a dropout layers with activation. Newly added layers are

```
1) Tanh activation layer
2) 20 nodes dense layer with relu activation
3) 30% dropout layer
4) Dense layer
5) last softmax layer
```


