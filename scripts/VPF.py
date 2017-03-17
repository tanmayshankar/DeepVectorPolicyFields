import numpy as npy
import matplotlib.pyplot as plt
from scipy import interpolate
import cv2
import copy
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline
# get_ipython().magic(u'matplotlib notebook')
cd ../../VectorFields

trans_model = npy.load("Data/Transition_Model.npy")
i = 7
policy = npy.load("Data/Policy_{0}.npy".format(i)).astype(int)
action_space = npy.array([[-1,0],[1,0],[0,-1],[0,1],[-1,-1],[-1,1],[1,-1],[1,1]])
discrete_size = 50


# In[4]:

N=50
Y,X = npy.mgrid[0:N,0:N]

U = npy.zeros(shape=(discrete_size,discrete_size))
V = npy.zeros(shape=(discrete_size,discrete_size))

for i in range(0,discrete_size):
	for j in range(0,discrete_size):
		U[i,j] = action_space[policy[i,j]][0]
		V[i,j] = action_space[policy[i,j]][1]


# In[5]:

fig, ax = plt.subplots()
im = ax.imshow(policy, origin='lower',extent=[-1,50,-1,50])

# ax.quiver(V,U)
ax.quiver(V,U,scale=1.5,width=0.3,units='xy')
fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()


# In[10]:

# for i in range(1,26):
#     pol = npy.loadtxt("Data/All_Used/reward_{0}/output_policy.txt".format(i))
#     npy.save("Policy_{0}.npy".format(i),pol)


# In[6]:

# THIS IS THE ACTUAL INTERPOLATION OF THE POLICY:
# X,Y = npy.mgrid[0:50,0:50]
X,Y = npy.mgrid[-25:25,-25:25]
test_range = npy.arange(-25,25,0.5)
print(len(test_range))
# policy_x = npy.zeros((50,50),dtype='int')
# policy_y = npy.zeros((50,50),dtype='int')
# for i in range(50):
#     for j in range(50):
#         policy_x[i,j] = action_space[policy[i,j],0]
#         policy_y[i,j] = action_space[policy[i,j],1]

policy[npy.where(policy==8.)]=0

func_x = interpolate.interp2d(X,Y,action_space[policy,0])
func_y = interpolate.interp2d(X,Y,action_space[policy,1])

# func_x = interpolate.interp2d(X,Y,policy_x)
# func_y = interpolate.interp2d(X,Y,policy_y)


# In[7]:

interp_policy_x = func_x(test_range,test_range)
interp_policy_y = func_y(test_range,test_range)


# In[ ]:




# In[8]:

resize_pol = copy.deepcopy(policy)
# resize_pol = npy.zeros((50,50),dtype=npy.float32)
resize_pol = resize_pol.astype(float)
fac = 2
# resize_pol = cv2.resize(resize_pol,(fac*resize_pol.shape[0],fac*resize_pol.shape[1]))
resize_pol = cv2.resize(resize_pol,(100,100))
resize_pol.shape


# In[9]:

fig, ax = plt.subplots()
# im = ax.imshow(policy, origin='lower',extent=[-1,50,-1,50])
# im = ax.imshow(policy, origin='lower',extent=[-25,25,-25,25])
im = ax.imshow(resize_pol,origin='lower')
# ax.quiver(V,U)
ax.quiver(interp_policy_x,interp_policy_y,scale=1.5,width=0.3,units='xy')
# ax.quiver(interp_policy_y,interp_policy_x,scale=1.2,width=0.5,units='xy')
fig.colorbar(im)
ax.set(aspect=1, title='Quiver Plot')
plt.show()


# In[26]:

policy3d = npy.zeros((3,3,3))

# space_range = npy.linspace(0,1,3)
# x,y,z=npy.meshgrid(space_range,space_range,space_range,indexing='ij')
# x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

# points = npy.zeros((3,3,3,3))
# for i in range(3):
#     for j in range(3):
#         for k in range(3):
#             points[i,j,k] = [i%2,j%2,k%2]

policy3d = npy.random.randint(0,6,(27))
print(policy3d)
action_list = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
actions = npy.array([action_list[i] for i in policy3d])
print(actions)

points = npy.zeros((27,3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            points[9*i+3*j+k] = [float(i)/2,float(j)/2,float(k)/2]


fig1 = plt.figure()

ax1 = fig1.gca(projection='3d')
ax1.scatter(points[:,0],points[:,1],points[:,2],c=npy.zeros(27),vmin=0,vmax=30,s=50)
# ax1.quiver(points[:,0],points[:,1],points[:,2],policy3d[:,0],policy3d[:,1],policy3d[:,2],pivot='tail')
ax1.quiver(points[:,0],points[:,1],points[:,2],actions[:,0],actions[:,1],actions[:,2],pivot='tail',length=0.35,)
ax1.scatter(sx,sy,sz,c=30,vmin=0,vmax=30,s=50)
# ax1.plot(points[:,0],points[:,1],points[:,2])


# In[16]:

[sx,sy,sz] = npy.random.random(3)
sx,sy,sz

kuhn = npy.zeros((6,4,3))

# In[ ]:
