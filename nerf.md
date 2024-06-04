# model- summary

 (x,y,z,theta, phi)->(r,g,b,sigma)

angle of ray-(theta,phi); density-(sigma)



(x,y,z)->FCN->sigma

(theta, phi) +sigma's previous layer output feature->FCN->(r,g,b)

# model

link: https://colab.research.google.com/github/bmild/nerf/blob/master/tiny_nerf.ipynb

`posenc` function:

> Embed position information into input features

input size: [n]

output size: [n*(2L+1)], L is L_embed



`init_model` function:

8 dense layer, width is 256

the activation function is relu



```python
    for i in range(D):
        outputs = dense()(outputs)
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs], -1)
```

every four layer, do residual connected.



input size:`[3 + 3*2*L_embed]`, 3-(x,y,z) 

output size: 4 - rgb, opacity



`get_rays` function:

> to get the rays direction

H-image height, W-image width,focal- Focal length of the camera,

c2w-The transformation matrix from the camera to the world coordinate system has a shape of (4, 4); The first three rows and the first three columns represent the rotation matrix; the last column of the first three rows represents the position of the camera in the world coordinate system

```python
dirs = tf.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -tf.ones_like(i)], -1)

```

i,j- the horizontal, vertical coordinate of every pixel, size-(H,W). then to get ray component on x,y,z axis. finally,dirs- the ray directions corresponds to every pixel,size-[H,W,3]



Then to convert from camera coordinates to world coordinates, we get

`rays_d` -the ray directions corresponds to every pixel,size-[H,W,3]



> to get the origin of each ray

`rays_o`-the same value since the camera position is same, size-[H,W,3]





`render_rays` function:

input: network_fn: nn model, rays_o, rays_d, near: near dist, far: far dist, N_samples: no of sample; rand: add rand or not



to compute 3d query points:

```python
pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
```

z_vals: `tf.linspace(near, far, N_samples)`, the depth of sample points, size-(H,W,N_samples)

pts- the 3d coordinates for all sample points of each ray, size-(H,W,N_samples,3)



to train and get opacities and colors

```python
    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    pts_flat = embed_fn(pts_flat)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])
    
    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3])
    rgb = tf.math.sigmoid(raw[...,:3]) #
```



to do volume rendering:

```python
    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1) #[H,W,n_samples]
    alpha = 1.-tf.exp(-sigma_a * dists) 
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    
    rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2) #[H,W,3]
```

since the formula:





[![image-20240604130928023.png](https://i.postimg.cc/SKMmyxwN/image-20240604130928023.png)](https://postimg.cc/sBskmsjt)



note that sigma_a is the density, alpha is the opacity, weights- alpha * cumulative transmission rate



calculate the loss of the true color and predicted color

# run on local machine(cpu)

too slow!!

[![QQ-20240603142114.jpg](https://i.postimg.cc/TwjGDHdz/QQ-20240603142114.jpg)](https://postimg.cc/ygNwqjSj)

[![QQ-20240603142351.jpg](https://i.postimg.cc/26Nr3fDd/QQ-20240603142351.jpg)](https://postimg.cc/hJsH36hh)

[![QQ-20240603142401.png](https://i.postimg.cc/4yJGbbnm/QQ-20240603142401.png)](https://postimg.cc/vgSCGnrs)









# generalize nerf

https://arxiv.org/pdf/2303.12786

to incorporate prior scene

![image-20240604121627803](C:\Users\lllps\AppData\Roaming\Typora\typora-user-images\image-20240604121627803.png)

I- image, f-image encoder, 

f(I)- image feature,  then  projected onto the image plane using pose to get f(I)_pi(x)



so that during test, one-pass manner is availble.

