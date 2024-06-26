# run nerf code with their data

train dataset:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXd6QfWz2HI2_PgjW3vMIg84KmT9_j3nL7dhVg1hia-PtcRe9qS1P15cMuXC_I96oN-qrfVULqpV7qVugjOO-zj2zQ_plMYiDeCdotMyWeFH_VtVpjGwAMHtc73OTH6z2zar10i5sWX3v-tf4u8Iv2DpCIQ?key=B1J_PdAqRoywV3PQfLu5hw)

 

 

result accuracy:

when iteration is 100000

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXdZ7W38YQ1brNLztHdVqtB4yegbIUPUdblOM7gAPeU3RhHEz2Aa4vPd3ClpUqGAAxExJpQLHDB6a2jdCWsmasADr3Ulrv0XCcvOXuGpIehtCpwwkN-I_cyfOxrUqb6zrHWqDldqJR6ayYPbyw2WPyHrHCIm?key=B1J_PdAqRoywV3PQfLu5hw)



result video:

https://drive.google.com/file/d/1AIhF7ZuL9I_YBS6Y0OBpgWVN97tZkfcf/view?usp=sharing



# run nerf using "feature-nerf" data

use bike dataset

modify code:

original train/val/test data size:100,13,25(8:1:2)

new train/val/test data size(50):(36:5:9)



result accuracy:

when iteration=2000(n_iterations = 2000,i_print = 50,i_img = 100,i_weights = 100,i_testset = 500,i_video = 500):

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXf431vCc4bIkPy58Yn79EWSeAZnX4-zg12TnGNKdj2NMchp25U5AIoi-UTzRdDRQuu9TUGAhnqgEbhG7tmc6vJn5QEujqacK5EfZtQ71rll2omUniCKuXmPzCoofPNWjgHtHurVlD3X1jkDtoBpq64PjrZi?key=B1J_PdAqRoywV3PQfLu5hw)

run successfully!



result video:





# GNFactor- MultiTask Real Robot Learning with Generalizable Neural Feature Fields

key point: 

A deep 3D voxel representation for simultaneous volume rendering and robotic motion prediction

pipeline：

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfwu-eIxZU17iTJjkcbIl7qqVp16hLjYG9tKbogytkGePRm8sRs4yKmCTBHaljXnUzDdTCK2QnIk-iZcX-Iey7roSXzOyOIaPJre9YIFaaWjfG-TpN3zT36K4UknckB7OqwlKweE0N0EpV5M4Vv9aYUkalk?key=B1J_PdAqRoywV3PQfLu5hw)





RGB-D-->Voxel Encoder-------------------------------------->volume render module(learn 3D respresentation and reconstructing RGB and 

​		(several 3D Convolutional Layers)                (Generalizable Neural Feature Field for reconstructed RGB and a pretrained VLM model(Stable Diffusion; 2D features are distilled into 3D voxel representations through a volume rendering module, using features extracted by pre-trained models as supervisory signals) for reconstructed vision-language embeddings

vision-language embeddings)------------------------->Action Prediction module(Perceiver Transformer)-------------->4Q values



> Perceiver Transformer:
>
> Input processing:
>
> 1.Downsample the 100³×128 3D volume representation to 20³×128 and flatten it into a sequence.
> 2.Project the robot state to 128 dimensions and combine it with the volume sequence.
> 3.Use the CLIP model to encode the language features into a 77×256 sequence.
> 4.Fuse the above features to form a 8077×256 sequence.
>
> Attention mechanism:
>
> 1.Add position encoding.
> 2.Process the sequence through 6 attention blocks containing self-attention and MLP layers.
>
> Output processing:
>
> 1.Restore the sequence to a 20³×256 voxel representation and upsample it to 100³×128.
>
> 2. Use 3D convolutional layers and MLP layers to predict action parameters such as translation, rotation, grasping, and collision avoidance										

​				

loss function:

Jointly training volumetric representations to meet the goals of reconstruction and motion prediction

Loss function for predicting actions:



![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXcDy-PSZAteG4rdMuvw3X7QBffmL_j7jnCxWjjITnH6P6EePLSi8174n4jlllKAaVTgCMCu62oQPSoSSarAZasmFI0EriViocDSiiOmWbIzCV0mPDcViDRiUKba-HxI3PpmrmpZZrH-h6QxtnIm5Zg_DqgC?key=B1J_PdAqRoywV3PQfLu5hw)

Vi = softmax(Qi) for Qi ∈ [Qtrans,Qopen,Qrot,Qcollide] and Yi ∈ [Ytrans,Yrot,Yopen,Ycollide] is the ground truth one-hot encoding. 



RGB reconstruction loss and feature reconstruction loss:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfd6f94IdziQ8Kk2C-CIhDxrI409ZO1v_T6v3QtnqtmHSvnw6ilB3Sjojts00RvsGu9QhJ5fwr4peR8uNmX6L1xfQs8GVKI3gSlkOyNHMzXDTdV8idyI6xbgZlzGulFMe6XvrxblxZCfrKjeG8-2TAS4Fg?key=B1J_PdAqRoywV3PQfLu5hw)





note that C(r) is ground truth color; F(r) is the ground truth vision-language embedding generated by Stable Diffusion

joint training:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfO-A5NXCaT2QXAbGFbhaqAqZQ5lk7pAqxv-ZW_j7OumeOR7QdUp9ubRssmtf77i2n3sSDUbyvKkSaQB3R8zfBmVIxQ4WxHDtrdy0pMD7hrZNusLKX4Yq7YhemRiA7wwS8zKrsyYSWswKsZcZQMuB8o93Qm?key=B1J_PdAqRoywV3PQfLu5hw)



# Learning Generalizable Feature Fields for Mobile Manipulation





pipeline:



![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXdjPYe_GKrH8x8pPTIfcwvNo_-tIOLvvKJCu99GYhgek0oS1aPcH-z0SQkO0xlJcdTF7iYIiIeWq_pqoJGkhwXgrDKOEHUZ_IyU1fkquhfQ_OSTOjyyKHiRfnKOK2i2EQxN4bHQ-9Vf7DeuYya4pdmunz3b?key=B1J_PdAqRoywV3PQfLu5hw)

f_enc(D):

besides GNFactor, add



SDF (Signed Distance Function) loss and Eikonal regularization are used to ensure that the geometric representation of the scene is accurate and smooth.



Depth Loss Better representation of scene geometry





loss function:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXelKEO4XaKRYCUNsYDDBMWHkKe561HlW5DrVygQzCUH7YPwVsPE-Dj9RTlzKPImIvmyvaRh7NRJHB99Ifahh-iXmCEe8PjB2BXHD2gfSbP9fiv_7UcurtB9H21Ntu5L83JFGi2NaIIY_RnyXHBCgAEN46aE?key=B1J_PdAqRoywV3PQfLu5hw)







# feature nerf

method:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfsdTKqV_8Gn09FFYue28bcTPfVgvsYLvWLPwyB9XaHkOs6ySk5goP0abnVWuR34rfXF_YZlecUxpPA8E-GEI2n4mLt7EfDvRPe48uCwZZCEaJpK69VweZ5D232pVpCiDZnHPU5E7UDXCkpB2LkbV7Nse-f?key=B1J_PdAqRoywV3PQfLu5hw)

1. generalized nerf:

   ![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXdngohuinGYHK_2NpTleHKUjiwKFaWAgUoRcY3YVa8xnSyT-pEIG1zntgU-aexv-O53Am2OR_semFM2JJmw0GMlCefHRpHU7RtHolMzgpk3FNDqWUbnnpJhs1DXS2w0z199H21WyQyMOk5xf1kXRuvkG2Jy?key=B1J_PdAqRoywV3PQfLu5hw)

   f- the image feature

2. Feature Distillation

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXcP7qS_fNq3aIUuRVjJU1Nx4HC271Tvs83DkxfeE6Vsl3q1xhlkuoXWxRhG1gxRQYfxvgu0qbjx2N-qiXapxFtISo1I0IDsDAOsBbC55F2RzeAsyuVaZ5AmWDgqif3Rl8foJDKjDo_QrNrpDSHfsjeIZFs?key=B1J_PdAqRoywV3PQfLu5hw)



v- feature vector

there exists f_teacher network that

the prediction of the feature vector is guided by the feature output of the teacher network



3.  Learning Internal NeRF Features for 3D Se mantic Understanding



![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXd-4js35_N5p-ouiJ0PDVU_Y0bMsQ_QiBiSoYhtYYKj-ZWb7GR5Xn0ivP6gc8oPdTHhGAN_0RLgFxtz0GoL18V7-zdJ_MDuVBmLIpEhV9vOnTA-A-gUVM6IoUYIbirnJBZe8zqVPYdjJ95CT_pHicdwTQQ2?key=B1J_PdAqRoywV3PQfLu5hw)





![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXcfny38MIxnpsXsNXxd2fx_lHWgS8MKJlYid7_TLwRmNutQq548gsotyboRX7BKQH_1Xpw4UGfr25mBDPtUCrDw4kYUqvsv5J9zCgmVz0MasURF_EkdDMgL4RSlSEQcCpEpoNyNFmajnYZPJFdBEpsz7Ek7?key=B1J_PdAqRoywV3PQfLu5hw)

Intermediate features representation: v_nerf

use a new mlp to predict the coordinate







loss function:

add 3 loss function:

rgb loss:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXcW1E5F4_TFl2qRlS8QIO6ZPqNaXrAjR819Jl97HMi2R9spBaKmYWR8CDVQ3zp32BZOL4rm8k41M5chsMECQVskHTmFFCDcNlkgClR4BkBxZ73mvyysz41ARkGxD7dhFYaUM-zHhQOzPOAgkh0v9u7Y2LMB?key=B1J_PdAqRoywV3PQfLu5hw)

C-real color

feature loss:

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfbEY1gk3WNQ6ZBUojfJ4dvVO_ixXmfPM9nwAcaSwMoPd2BS4zA-APIDADZOBgGMbH75h4fDDYadTA_LT2K3ILAkxXgffFgKShsp2CCcGUxBgiSUzSuWYlmmJUJNJxbO9cS7dPkXFVHR_6cOCtvoenhTKOd?key=B1J_PdAqRoywV3PQfLu5hw)

V - the pretrained teacher network output

3D Se mantic loss:



![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXdCM-cFu-xQLgjR-sWeoIJqgT1yPYfU0IlQdDJH67ucYQxJGntmyCKD2o4U9nNXjMMS-sijB61RptQZFbB2RTg134A3gsfHGOcRQYUjFeXK11PGkmChNiM-M0kpqQV7oCHIJMvEuzvHPAOFe452Aja8nzvV?key=B1J_PdAqRoywV3PQfLu5hw)

r-real coordinates

 L = Lrec + distillLdistill + coordLcoord

