# dataset looks like

data["image"].keys(): 'corner1', 'corner2', 'corner3', 'corner4', 'gripperPOV'

data["image"] ["corner1"].len = 125

data["image"] ["corner1"] [0]

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXdHSsW0aTaKRQDgNgzwmGqbpvwMpq2DPlczA6vaFZubFBiOMpVKrU4cO1FIg8jk-oyPkFNG67ZFEJzoBUee3qBtP9A3pQRgHhavnnA5Yn4CWMzLngJO8V1bmnuNv-BC2uGKWK2QgXiTK0g9jevWoZN1-Nde?key=UzChH-Eo39ry3AOSYq8HZA)

data["image"] ["corner1"] [0].shape: (3, 84, 84)

data["image"] ["corner1"] [124]

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfq5REs_WIOd7gmBZHVn86KwoKwRbKzFCH9H2vswVlPADEUqojnPgE3U8fzVuQxAeLAvVmZYhksVennyAgMdedm_svejjeqwgQ7XUWPaxG0v00QNQeYsLzOXVAqOMNyNqL9pBQ9mWXKug3HjKn0_l6GMSZj?key=UzChH-Eo39ry3AOSYq8HZA)

data["image"] ["corner2"] [0]

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXe9qlRPlvu8c7g-gn4LWb3D_WjzjaLEbsnq2gafEXKvJ78siaNjG92mXDJZHWq_ZBspsL7b8MwsaxiUvxXkCYYqj3P3A8InXzLe4GlhvuPKD-ClOejIIio9OalR5A7Kh9XAKTc7eIav0Y67xcsCErDt0Xwe?key=UzChH-Eo39ry3AOSYq8HZA)



data["image"] ["corner2"] [124]



![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXcjbiSK8R_LpvOI4Pc_b5vOa0Kg0j96irP62ow3fffgvh3t8mq0SexEUhaC3KOoIeWGp6wInagNC8GjJXwXgV8OhdkvKWES9aJtT9Cf3tYkxsm54M5ufB7JGTuEqno_uEd1ja-HFkdYsiOxjwRq0F8A9AY?key=UzChH-Eo39ry3AOSYq8HZA)



data ["point_cloud"] [0] ['camera_poses']: 'corner1', 'corner2', 'corner3', 'corner4', 'gripperPOV'

data ["point_cloud"] [0] ['camera_poses'] ['corner1'].shape:(4,4)



data ['info'] ['fovy'] ['corner1']

45.0 ------- fovy_radians: 0.7853981633974483



data ['info'] ['camera_intrinsics'] ['corner1']

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXerQcOdc2nKYEquzuYfAi_Wl4BcBuZy-sotkwDawfBeXq7S_9UMMJe9N3ZE9GfTonqsfVjUW_1qtLeXy55VByWF7X1SbM8j7ny8jCIB9YQgAtqk1i4jIDhaKWIOEXjvCqcazd--mY2JCweedUDBKhvnHdw?key=UzChH-Eo39ry3AOSYq8HZA)

so principal point is the middle of the camera









# new dataset 

train on 5 frames



configuation:

white_bkgd: false

half_res:false



i_img:500->400000


i_testset:50000->400000


i_video:50000->400000


first try:

N_rand = 1024->256

N_samples = 64->32

N_importance = 128->64

precrop_iters = 500->50
N_iters = 200000+1->10000 + 1

the results:
![image](https://github.com/lllpsy/rl-lab/assets/59329407/7eb7bf71-b610-4e0f-9ed2-34e27dc50bcb)

[TRAIN] Iter: 10000 Loss: 0.001231816248036921  PSNR: 31.56230926513672   


second try:

N_rand = 1024->256

N_iters = 200000+1->100000 + 1





# dataset of paper

begin: (loss psnr)

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXeJpqQp_tIIwjVIYX6U9wTh4You7DtTN5_MIt12Y24ltERNy3CuQsVtnoIKwjg9X4jDG_s8SrrOZkit6Fgh_Wat8gVdZQLhH3cOpQUv_eHUAs51uL2566wwkAUcrZfTI6GHQlPPGGBkFq1d-07EPg1sQDL3?key=UzChH-Eo39ry3AOSYq8HZA)

when iteration is 10000:
![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXeR8-x4vQywK8nPZmG28UOJmuM74AypHZa6zSylZz3vrb-ms2-ihecrtZpHYefcxhQFiqglYMN-CHQHvFaEx87Al2jB-QV-hGAnKAbvy4AGgIDtZFOzmFEgmkBgdfKepb8n7nQnSZMhNn8wjUNNXCAf5ngC?key=UzChH-Eo39ry3AOSYq8HZA)



when iteration is 100000:

![img](https://camo.githubusercontent.com/25a5f2c512d0e8d4e74a5eb8dbbbf4f6ef702ec122c13208d48e06a1e2df0812/68747470733a2f2f6c68372d75732e676f6f676c6575736572636f6e74656e742e636f6d2f646f63737a2f41445f346e58645a3757333859513162724e4c7a7448645671744234796567624955505564626c4f4d37674150655533526848457a3241613476506433436c7055714741417845784a70514c4844423661326a644357736d617341447233556c727630584363764f5875477049656874437077776b4e2d495f6379664f7872557162367a72485771446c64714a5236617959506279773257507948724843496d3f6b65793d42314a5f50644171526f797756335051664c75356877)





train in smaller iteration(fast time!!-1 day)
lrate: 5e-4-> 1e-3

i_img(frequency of tensorboard image logging):500->50

i_print(ffrequency of console printout and metric loggin):100->10

i_weights:10000->500

i_testset:50000->2500

i_video:50000->2500



  N_iters = 200000+1->10000 + 1



![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXfnSgzCOsBsSFKn3ynrJFV8oM0cjXQENQiTK7GLKC5jMSk9QIWJ8jYEhFnV44lzA7uHeTuUh4yKuWZTuh_LFiTHytjvOC2xxsGOnQ65UIZBQL-JYQAxp6x_o1cMuQXZvyz6sgPkDQHF-5GTuGpLoNY-z-_9?key=UzChH-Eo39ry3AOSYq8HZA)

![img](https://lh7-us.googleusercontent.com/docsz/AD_4nXf8lBnrypZfIMxUzha68hjHCJwMgignpRr7ZHnVcrfMhtEpmV8nkbghxI41Qixsg-FxX8UdlZq5Rl4bS1vhvRDGr9CvnMfLF1HbQ1RNYK1HIHSxyFPGU_VoteVN8-IoZjVSl8YKesBjl90HL_RT0yFuo3sM?key=UzChH-Eo39ry3AOSYq8HZA)

it seems that the result looks well 

the generated video:

https://drive.google.com/file/d/1AIhF7ZuL9I_YBS6Y0OBpgWVN97tZkfcf/view?usp=sharing
