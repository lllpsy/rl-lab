

# how the camera info influence the res?

> nerf:  use c2w to construct 3d rays and generate image
>
> gs: use w2c to change 3d gaussians to 2d and generate image 



https://github.com/SY-007-Research/3dgs_render_python



how to achieve gs:

in guassian_render.py:

first step:

compute 3d covarance by scaling and rotation parameters

```python
def computeCov3D(scale, mod, rot):
    # create scaling matrix
    S = np.array(
        [[scale[0] * mod, 0, 0], [0, scale[1] * mod, 0], [0, 0, scale[2] * mod]]
    )

    # normalize quaternion to get valid rotation
    # we use rotation matrix
    R = rot

    # compute 3d world covariance matrix Sigma
    M = np.dot(R, S)
    cov3D = np.dot(M, M.T)

    return cov3D
```



step2:

compute 2D screen-space covariance matrix

```python
def computeCov2D(mean, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix):

    t = transformPoint4x3(mean, viewmatrix)

    limx = 1.3 * tan_fovx
    limy = 1.3 * tan_fovy
    txtz = t[0] / t[2]
    tytz = t[1] / t[2]
    t[0] = min(limx, max(-limx, txtz)) * t[2]
    t[1] = min(limy, max(-limy, tytz)) * t[2]

    J = np.array(
        [
            [focal_x / t[2], 0, -(focal_x * t[0]) / (t[2] * t[2])],
            [0, focal_y / t[2], -(focal_y * t[1]) / (t[2] * t[2])],
            [0, 0, 0],
        ]
    )
    W = viewmatrix[:3, :3]
    T = np.dot(J, W)

    cov = np.dot(T, cov3D)
    cov = np.dot(cov, T.T)

    # Apply low-pass filter
    # Every Gaussia should be at least one pixel wide/high
    # Discard 3rd row and column
    cov[0, 0] += 0.3
    cov[1, 1] += 0.3
    return [cov[0, 0], cov[0, 1], cov[1, 1]]
```

Q:

1. to obtain focal_x:

   viewpoint_camera.FoVx

   ```python
   fovx = contents["camera_angle_x"]
   
   
   fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
   FovY = fovy 
   FovX = fovx
   
   
   def fov2focal(fov, pixels):
       return pixels / (2 * math.tan(fov / 2))
   
   def focal2fov(focal, pixels):
       return 2*math.atan(pixels/(2*focal))
   ```

   

   in rasterizer_impl.cu:

   ```c++
   const float focal_y = height / (2.0f * tan_fovy);
   const float focal_x = width / (2.0f * tan_fovx);
   ```

   

   

2. to obtain tan_fovx:

   in gaussian_render.py:

   ```python
   tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
   tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
   ```



step3: use gaussains to represent the color

```python
# pos - mean ofgaussian
# campos- campos=viewpoint_camera.camera_center
#sh- 16*3 mat
def computeColorFromSH(deg, pos, campos, sh):

    dir = pos - campos
    dir = dir / np.linalg.norm(dir)

    # color when not considering view dirs
    result = SH_C0 * sh[0]

    if deg > 0:
        x, y, z = dir
        result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3]

        if deg > 1:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z
            result = (
                result
                + SH_C2[0] * xy * sh[4]
                + SH_C2[1] * yz * sh[5]
                + SH_C2[2] * (2.0 * zz - xx - yy) * sh[6]
                + SH_C2[3] * xz * sh[7]
                + SH_C2[4] * (xx - yy) * sh[8]
            )

            if deg > 2:
                result = (
                    result
                    + SH_C3[0] * y * (3.0 * xx - yy) * sh[9]
                    + SH_C3[1] * xy * z * sh[10]
                    + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[11]
                    + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[12]
                    + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[13]
                    + SH_C3[5] * z * (xx - yy) * sh[14]
                    + SH_C3[6] * x * (xx - 3.0 * yy) * sh[15]
                )
    result += 0.5
    return np.clip(result, a_min=0, a_max=1)
```

Q: viewpoint_camera.camera_center?

in scene/dataset_readers.py

```python
# NeRF 'transform_matrix' is a camera-to-world transform
c2w = np.array(frame["transform_matrix"])

w2c = np.linalg.inv(pose.numpy())
R = np.transpose(w2c[:3, :3])
T = w2c[:3, 3]


self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
    
self.camera_center = self.world_view_transform.inverse()[3, :3]
```



step4: to render

we have model output color and opacity

use these to compute alpha, then transmission,

similar with nerf, to get the generated image color

```python
    def render(
        self, point_list, W, H, points_xy_image, features, conic_opacity, bg_color
    ):

        out_color = np.zeros((H, W, 3))
        pbar = tqdm(range(H * W))

        # loop pixel
        for i in range(H):
            for j in range(W):
                pbar.update(1)
                pixf = [i, j]
                C = [0, 0, 0]

                # loop gaussian
                for idx in point_list:

                    # init helper variables, transmirrance
                    T = 1

                    # Resample using conic matrix
                    # (cf. "Surface Splatting" by Zwicker et al., 2001)
                    xy = points_xy_image[idx]  # center of 2d gaussian
                    d = [
                        xy[0] - pixf[0],
                        xy[1] - pixf[1],
                    ]  # distance from center of pixel
                    con_o = conic_opacity[idx]
                    power = (
                        -0.5 * (con_o[0] * d[0] * d[0] + con_o[2] * d[1] * d[1])
                        - con_o[1] * d[0] * d[1]
                    )
                    if power > 0:
                        continue

                    # Eq. (2) from 3D Gaussian splatting paper.
                    # Compute color
                    alpha = min(0.99, con_o[3] * np.exp(power))
                    if alpha < 1 / 255:
                        continue
                    test_T = T * (1 - alpha)
                    if test_T < 0.0001:
                        break

                    # Eq. (3) from 3D Gaussian splatting paper.
                    color = features[idx]
                    for ch in range(3):
                        C[ch] += color[ch] * alpha * T

                    T = test_T

                # get final color
                for ch in range(3):
                    out_color[j, i, ch] = C[ch] + T * bg_color[ch]

        return out_color
```

Q: the difference to nerf? faster-gpu, 16*16 sections, 1 gpu run 1 section to calculate corresponding gaussians

1. no need to calculate the ray points 2. need to sort the guassains according to z depth









# Q:change c2w into w2c?

no image





# Observe the video changes as the number of iterations increases

when render_video == 5000

output/6aa0c90e-3

when iteration == 5000

https://drive.google.com/file/d/15zW6fpRz_84Gwp4PQYmU6G6EHhvaQcZK/view?usp=drive_link



when render_video==500

output/0f629cfb-f

when iteration == 500

https://drive.google.com/file/d/1oPJgO-NoazLNELgKLgdFgiwQwTdInYDI/view?usp=drive_link



when iteration == 1000

https://drive.google.com/file/d/1baNQzv1Cj3314uBm2oXKHHs1vtC0gx1D/view?usp=drive_link

