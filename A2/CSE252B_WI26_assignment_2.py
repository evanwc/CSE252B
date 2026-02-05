# %% [markdown]
# # CSE 252B: Computer Vision II, Winter 2026 – Assignment 2
# 

# %% [markdown]
# Instructor: Ben Ochoa
# 
# Assignment due: Wed, Feb 4, 11:59 PM

# %% [markdown]
# **Name:** Evan Cheng
# 
# **PID:** A69042831

# %% [markdown]
# ## Instructions
# * Review the academic integrity and collaboration policies on the course 
# website.
# * This assignment must be completed individually.
# * All solutions must be written in this notebook.
# * Math must be done in Markdown/$\LaTeX$.
# * You must show your work and describe your solution.
# * Programming aspects of this assignment must be completed using Python in this notebook.
# * Your code should be well written with sufficient comments to understand, but there is no need to write extra markdown to describe your solution if it is not explictly asked for.
# * This notebook contains skeleton code, which should not be modified (this is important for standardization to facilate efficient grading).
# * You may use python packages for basic linear algebra, but you may not use functions that directly solve the problem. If you are uncertain about using a specific package, function, or method, then please ask the instructional staff whether it is allowable.
# * **You must submit this notebook as an .ipynb file, a .py file, and a .pdf file on Gradescope.**
#     - You may directly export the notebook as a .py file.  You may use [nbconvert](https://nbconvert.readthedocs.io/en/latest/install.html) to convert the .ipynb file to a .py file using the following command
#     `jupyter nbconvert --to script filename.ipynb`
#     - There are two methods to convert the notebook to a .pdf file.
#         - You may first export the notebook as a .html file, then print the web page as a .pdf file.
#         - If you have XeTeX installed, then you may directly export the notebook as a .pdf file.  You may use [nbconvert](https://nbconvert.readthedocs.io/en/latest/install.html) to convert a .ipynb file to a .pdf file using the following command
#         `jupyter nbconvert --allow-chromium-download --to webpdf filename.ipynb`
#     - **You must ensure the contents in each cell (e.g., code, output images, printed results, etc.) are clearly visible, and are not cut off or partially cropped in the .pdf file.**
#     - Your code and results must remain inline in the .pdf file (do not move your code to an appendix).
#     - **While submitting on gradescope, you must assign the relevant pages in the .pdf file submission for each problem.**
# * It is highly recommended that you begin working on this assignment early.

# %% [markdown]
# ## Problem 1 (Math): Line-plane intersection (5 points)
#   The line in 3D defined by the join of the points $\boldsymbol{\mathrm{X}}_1 = 
#   (X_1, Y_1, Z_1, T_1)^\top$ and $\boldsymbol{\mathrm{X}}_2 = (X_2, Y_2, Z_2, 
#   T_2)^\top$ can be represented as a Plücker matrix $\mathtt{L} = 
#   \boldsymbol{\mathrm{X}}_1 \boldsymbol{\mathrm{X}}_2^\top - 
#   \boldsymbol{\mathrm{X}}_2 \boldsymbol{\mathrm{X}}_1^\top$ or pencil of points 
#   $\boldsymbol{\mathrm{X}}(\lambda) = \lambda \boldsymbol{\mathrm{X}}_1 + (1 - 
#   \lambda) \boldsymbol{\mathrm{X}}_2$ (i.e., $\boldsymbol{\mathrm{X}}$ is a 
#   function of $\lambda$).  The line intersects the plane 
#   $\boldsymbol{\mathrm{\pi}} = (a, b, c, d)^\top$ at the point 
#   $\boldsymbol{\mathrm{X}}_\mathtt{L} = \mathtt{L} \boldsymbol{\mathrm{\pi}}$ 
#   or $\boldsymbol{\mathrm{X}}(\lambda_{\boldsymbol{\mathrm{{\pi}}}})$, where 
#   $\lambda_{\boldsymbol{\mathrm{\pi}}}$ is determined such that 
#   $\boldsymbol{\mathrm{X}}(\lambda_{\boldsymbol{\mathrm{\pi}}})^\top 
#   \boldsymbol{\mathrm{\pi}} = 0$ (i.e., 
#   $\boldsymbol{\mathrm{X}}(\lambda_{\boldsymbol{\mathrm{\pi}}})$ is the point 
#   on $\boldsymbol{\mathrm{\pi}}$).  Show that 
#   $\boldsymbol{\mathrm{X}}_\mathtt{L}$ is equal to 
#   $\boldsymbol{\mathrm{X}}(\lambda_{\boldsymbol{\mathrm{\pi}}})$ up to scale.

# %% [markdown]
# Intersection of line and plane:
# $
# \\
# \mathrm{X}_\mathtt{L} = \mathtt{L}\pi  = (\mathrm{X}_1\mathrm{X}_2^\top - \mathrm{X}_2\mathrm{X}_1^\top)\pi = \mathrm{X}_1(\mathrm{X}_2^\top\pi) - \mathrm{X}_2(\mathrm{X}_1^\top\pi)
# \\[0.5cm]
# $
# Substituting $X(\lambda)$ with $X(\lambda_\pi^\top\pi)$:
# $
# \\
# \mathrm{X}(\lambda) = \lambda\mathrm{X}_1 + (1 - \lambda) \mathrm{X}_2
# \\
# \mathrm{X}(\lambda_\pi)^\top\pi = 0 = \lambda_\pi(\mathrm{X}_1^\top\pi) + (1 - \lambda_\pi)(\mathrm{X}_2^\top\pi) = \lambda_\pi(\mathrm{X}_1^\top\pi) + \mathrm{X}_2^\top\pi - \lambda_\pi(\mathrm{X}_2^\top\pi) = \lambda_\pi(\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi) + \mathrm{X}_2^\top\pi 
# \\[0.5cm]
# $
# Coefficients:
# $
# \\
# \lambda_\pi = \frac{-\mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} 
# \\
# 1 - \lambda_\pi = 1 - \frac{-\mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} = \frac{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} - \frac{-\mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} = \frac{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} + \frac{\mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} = \frac{\mathrm{X}_1^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi} 
# \\[0.5cm]
# $
# Plug back in:
# $
# \\
# \mathrm{X}(\lambda_\pi) = \lambda_\pi\mathrm{X}_1 + (1 - \lambda)\mathrm{X}_2 
# \\
# \mathrm{X}(\lambda_\pi) = \frac{-\mathrm{X}_2^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi}\mathrm{X}_1 + \frac{\mathrm{X}_1^\top\pi}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi}\mathrm{X}_2 = \frac{1}{\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi}(-\mathrm{X}_2^\top\pi\mathrm{X}_1 + \mathrm{X}_1^\top\pi\mathrm{X}_2) 
# \\
# -(\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi)\mathrm{X}(\lambda_\pi) = \mathrm{X}_2^\top\pi\mathrm{X}_1 - \mathrm{X}_1^\top\pi\mathrm{X}_2 = \mathrm{X}_1(\mathrm{X}_2^\top\pi) - \mathrm{X}_2(\mathrm{X}_1^\top\pi) = \mathrm{X}_\mathtt{L} 
# \\
# -(\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi)\mathrm{X}(\lambda_\pi) = \mathrm{X}_\mathtt{L}
# \\[0.5cm]
# $
# $-(\mathrm{X}_1^\top\pi - \mathrm{X}_2^\top\pi)$ is a scalar, therefore:
# $
# \\
# \mathrm{X}(\lambda_\pi) = \mathrm{X}_\mathtt{L}
# $ up to scale

# %% [markdown]
# ## Problem 2 (Math): Line-quadric intersection (5 points)
#   In general, a line in 3D intersects a quadric $\mathtt{Q}$ at zero, one (if 
#   the line is tangent to the quadric), or two points.  If the pencil of points 
#   $\boldsymbol{\mathrm{X}}(\lambda) = \lambda \boldsymbol{\mathrm{X}}_1 + (1 -
#   \lambda) \boldsymbol{\mathrm{X}}_2$ represents a line in 3D, the (up to two) 
#   real roots of the quadratic polynomial $c_2 \lambda_{\mathtt{Q}}^2 + c_1
#   \lambda_{\mathtt{Q}} + c_0 = 0$ are used to solve for the intersection
#   point(s) $\boldsymbol{\mathrm{X}}(\lambda_{\mathtt{Q}})$.  Show that $c_2 =
#   \boldsymbol{\mathrm{X}}_1^\top \mathtt{Q} \boldsymbol{\mathrm{X}}_1 - 2 
#   \boldsymbol{\mathrm{X}}_1^\top \mathtt{Q} \boldsymbol{\mathrm{X}}_2 + 
#   \boldsymbol{\mathrm{X}}_2^\top \mathtt{Q} \boldsymbol{\mathrm{X}}_2$, $c_1 = 
#   2 (\boldsymbol{\mathrm{X}}_1^\top \mathtt{Q} \boldsymbol{\mathrm{X}}_2 - 
#   \boldsymbol{\mathrm{X}}_2^\top \mathtt{Q} \boldsymbol{\mathrm{X}}_2 )$, and 
#   $c_0 = \boldsymbol{\mathrm{X}}_2^\top \mathtt{Q} \boldsymbol{\mathrm{X}}_2$.

# %% [markdown]
# Substituting $\mathrm{X}$ with $\mathrm{X}(\lambda_\mathtt{Q})$:
# $
# \\
# \mathrm{X}^\top\mathtt{Q}\mathrm{X} = 0
# \\
# \mathrm{X}(\lambda_\mathtt{Q})^\top\mathtt{Q}\mathrm{X}(\lambda_\mathtt{Q}) = 0
# \\[0.5cm]
# $
# Substituting pencil of points:
# $
# \\
# (\lambda_\mathtt{Q}\mathrm{X}_1 + (1 - \lambda_\mathtt{Q})\mathrm{X}_2)^\top\mathtt{Q}(\lambda_\mathtt{Q}\mathrm{X}_1 + (1 - \lambda_\mathtt{Q})\mathrm{X}_2) = 0
# \\
# (\lambda_\mathtt{Q}\mathrm{X}_1 + \mathrm{X}_2 - \lambda_\mathtt{Q}\mathrm{X}_2)^\top\mathtt{Q}(\lambda_\mathtt{Q}\mathrm{X}_1 + \mathrm{X}_2 - \mathrm{X}_2\lambda_\mathtt{Q}) = 0
# \\
# (\lambda_\mathtt{Q}\mathrm{X}_1^\top + \mathrm{X}_2^\top - \lambda_\mathtt{Q}\mathrm{X}_2^\top)\mathtt{Q}(\lambda_\mathtt{Q}\mathrm{X}_1 + \mathrm{X}_2 - \mathrm{X}_2\lambda_\mathtt{Q}) = 0
# \\
# \lambda_\mathtt{Q}^2\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_1 + \lambda_\mathtt{Q}\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 - \lambda_\mathtt{Q}^2\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 + \mathrm{X}_2^\top\mathtt{Q}\lambda_\mathtt{Q}\mathrm{X}_1 + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2 - \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2\lambda_\mathtt{Q} - \lambda_\mathtt{Q}^2\mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_1 - \lambda_\mathtt{Q}\mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2 + \lambda_\mathtt{Q}^2\mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2 = 0
# \\
# \lambda_\mathtt{Q}^2(\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_1 - \mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 - \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_1 + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2) + \lambda_\mathtt{Q}(\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_1 - 2\mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2) + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2 = 0
# \\[0.5cm]
# $
# Since $\mathtt{Q}$ is symmetric:
# $
# \\
# \lambda_\mathtt{Q}^2(\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_1 - 2\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2) + \lambda_\mathtt{Q}(2(\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 - \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2)) + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2 = 0
# \\[0.5cm]
# $
# Therefore:
# $
# \\
# c_2 = \mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_1 - 2\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 + \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2
# \\
# c_1 = 2(\mathrm{X}_1^\top\mathtt{Q}\mathrm{X}_2 - \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2)
# \\
# c_0 = \mathrm{X}_2^\top\mathtt{Q}\mathrm{X}_2
# $

# %% [markdown]
# ## Problem 3 (Programming):  Linear Estimation of the Camera Projection Matrix (15 points)
#   Download input data from the course website.  The file `hw2_points3D.txt` 
#   contains the coordinates of 50 scene points in 3D (each line of the file 
#   gives the $\tilde{X}_i$, $\tilde{Y}_i$, and $\tilde{Z}_i$ inhomogeneous 
#   coordinates of a point).  The file `hw2_points2D.txt` contains the 
#   coordinates of the 50 corresponding image points in 2D (each line of the file 
#   gives the $\tilde{x}_i$ and $\tilde{y}_i$ inhomogeneous coordinates of a 
#   point).  The scene points have been randomly generated and projected to image 
#   points under a camera projection matrix (i.e., $\boldsymbol{\mathrm{x}}_i = 
#   \mathtt{P} \boldsymbol{\mathrm{X}}_i$), then noise has been added to the 
#   image point coordinates.
# 
#   Estimate the camera projection matrix $\mathtt{P}_\text{DLT}$ using the 
#   direct linear transformation (DLT) algorithm (with data normalization).  You 
#   must express $\boldsymbol{\mathrm{x}}_i = \mathtt{P} 
#   \boldsymbol{\mathrm{X}}_i$ as $[\boldsymbol{\mathrm{x}}_i]^\perp \mathtt{P} 
#   \boldsymbol{\mathrm{X}}_i = \boldsymbol{0}$ (not $\boldsymbol{\mathrm{x}}_i 
#   \times \mathtt{P} \boldsymbol{\mathrm{X}}_i = \boldsymbol{0}$), where
#   $[\boldsymbol{\mathrm{x}}_i]^\perp \boldsymbol{\mathrm{x}}_i = 
#   \boldsymbol{0}$, when forming the solution. Return $\mathtt{P}_\text{DLT}$, 
#   scaled such that $\lVert\mathtt{P}_\text{DLT}\rVert_\text{Fro} = 1$
#   
#   The following helper functions may be useful in your DLT function 
#   implementation.  You are welcome to add any additional helper functions.

# %%
import numpy as np
import time

def homogenize(x):
    # Converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))


def dehomogenize(x):
    # Converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]


def data_normalize(pts):
    # Data normalization of n dimensional pts
    #
    # Input:
    #    pts - points data in inhomogeneous coordinates
    # Outputs:
    #    pts - normalized points data in homogeneous coordinates
    #    T - corresponding transformation matrix

    """your code here"""
    n = pts.shape[0]

    s = np.sqrt(n / np.sum(np.var(pts, axis=1)))
    mu = np.mean(pts, axis=1)

    T = np.eye(pts.shape[0]+1)
    T[:n,:n] *= s
    T[:n,n] = -s * mu
    
    pts = homogenize(pts)
    pts = T @ pts

    return pts, T

def sum_of_square_projection_error(P, x, X):
    # Computes the sum of squares of the reprojection error
    # Inputs:
    #    P - the camera projection matrix
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    # Output:
    #    cost - Sum of squares of the reprojection error

    """your code here"""
    X_h = homogenize(X)
    x_proj = dehomogenize(P @ X_h)
    errors = x - x_proj

    cost = np.sum(errors**2)
    return cost

# %%
def estimate_camera_projection_matrix_linear(x, X, normalize=True):
    # Inputs:
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    normalize - if True, apply data normalization to x and X
    #
    # Output:
    #    P - the (3x4) DLT estimate of the camera projection matrix
    P = np.eye(3,4)+np.random.randn(3,4)/10

    # data normalization
    if normalize:
        x, T = data_normalize(x)
        X, U = data_normalize(X)
    else:
        x = homogenize(x)
        X = homogenize(X)

    """your code here"""
    e_1 = np.array([1,0,0])

    A = []

    for i in range(x.shape[1]):
        x_i = x[:,i]
        X_i = X[:,i]

        v = x_i + np.sign(x_i[0]) * np.linalg.norm(x_i) * e_1
        v = v.reshape((3, 1))
        
        H_v = np.eye(x.shape[0]) - 2 * ((v @ v.T) / (v.T @ v))
        x_left = H_v[1:, :]

        A.append(np.kron(x_left, X_i.T))

    A = np.vstack(A)

    _, _, V_T = np.linalg.svd(A)
    P = V_T[-1].reshape(3, 4)

    # data denormalize
    if normalize:
        P = np.linalg.inv(T) @ P @ U

    return P

def display_results(P, x, X, title):
    print(title+' =')
    print (P/np.linalg.norm(P)*np.sign(P[-1,-1]))

# load the data
x=np.loadtxt('hw2_points2D.txt').T
X=np.loadtxt('hw2_points3D.txt').T

assert x.shape[1] == X.shape[1]
n = x.shape[1]

# compute the linear estimate without data normalization
print ('Running DLT without data normalization')
time_start=time.time()
P_DLT = estimate_camera_projection_matrix_linear(x, X, normalize=False)
cost = sum_of_square_projection_error(P_DLT, x, X)
time_total=time.time()-time_start
# display the results
print('took %f secs'%time_total)
print('Cost=%.9f'%cost)


# compute the linear estimate with data normalization
print ('Running DLT with data normalization')
time_start=time.time()
P_DLT = estimate_camera_projection_matrix_linear(x, X, normalize=True)
cost = sum_of_square_projection_error(P_DLT, x, X)
time_total=time.time()-time_start
# display the results
print('took %f secs'%time_total)
print('Cost=%.9f'%cost)

print("\n==Correct outputs==")
print("Cost=%.9f without data normalization"%97.053718991)
print("Cost=%.9f with data normalization"%84.104680130)

# %%
# Report your P_DLT (estimated camera projection matrix linear) value here!
display_results(P_DLT, x, X, 'P_DLT')

# %% [markdown]
# ## Problem 4 (Programming):  Nonlinear Estimation of the Camera Projection Matrix (30 points)
#   Use $\mathtt{P}_\text{DLT}$ as an initial estimate to an iterative estimation 
#   method, specifically the Levenberg-Marquardt algorithm, to determine the 
#   Maximum Likelihood estimate of the camera projection matrix that minimizes 
#   the projection error.  You must parameterize the camera projection matrix as 
#   a parameterization of the homogeneous vector $\boldsymbol{\mathrm{p}} = 
#   \text{vec}(\mathtt{P}^\top)$.  It is highly recommended to implement a 
#   parameterization of homogeneous vector method where the homogeneous vector is 
#   of arbitrary length, as this will be used in following assignments.
#   
#   Report the initial cost (i.e., cost at iteration 0) and the cost at the end
#   of each successive iteration. Show the numerical values for the final 
#   estimate of the camera projection matrix $\mathtt{P}_\text{LM}$, scaled such 
#   that $\lVert\mathtt{P}_\text{LM}\rVert_\text{Fro} = 1$.
#   
#   The following helper functions may be useful in your LM function 
#   implementation. You are welcome <i>and encouraged</i> to add any additional 
#   helper functions.
#   
#   Hint: LM has its biggest cost reduction after the first iteration. If you do 
#   not experience this, then there may be an issue with your implementation.

# %%
# Note that np.sinc is different than defined in class
def sinc(x):

    # Returns a scalar valued sinc value
    """your code here"""
    if x == 0: 
        y = 1
    else:
        y = np.sin(x) / x

    return y

def d_sinc(x):
    # Returns a scalar valued derivative of sinc value

    """your code here"""
    if x == 0:
        y = 0
    else:
        y = (np.cos(x) / x) - (np.sin(x) / x**2)

    return y

def partial_x_partial_p(P,X,x):
    # Compute the dx_dp component for the Jacobian
    #
    # Input:
    #    P - 3x4 projection matrix
    #    X - Homogenous 3D scene point
    #    x - inhomogenous 2D point
    # Output:
    #    dx_dp - 2x12 matrix

    dx_dp = np.zeros((2,12))

    """your code here"""
    w = P[2,:] @ X
    X_T = X.reshape(1, 4)

    dx_dp[0,:] = np.hstack([X_T, np.zeros((1, 4)), -x[0] * X_T]) / w
    dx_dp[1,:] = np.hstack([np.zeros((1, 4)), X_T, -x[1] * X_T]) / w
    return dx_dp


def parameterize_matrix(P):
    # Wrapper function to interface with LM.
    # Takes all optimization variables and parameterizes all of them.
    # In this case it is just P, but in future assignments it will be more useful.
    return parameterize_homog(P.reshape(-1,1))


def deparameterize_matrix(m,rows,columns):
    # Deparameterize all optimization variables
    # Input:
    #   m - matrix to be deparameterized
    #   rows - number of rows of the deparameterized matrix
    #   columns - number of rows of the deparameterized matrix
    #
    # Output:
    #    deparameterized matrix
    #
    # For the camera projection, deparameterize it using deparameterize_matrix(p,3,4)
    # where p is the parameterized camera projection matrix

    return deparameterize_homog(m).reshape(rows,columns)


def parameterize_homog(v_bar):
    # Given a homogeneous vector v_bar return its minimal parameterization
    """your code here"""
    a = v_bar[0]
    b = v_bar[1:]

    v = 2 / (sinc(np.arccos(a))) * b

    norm = np.linalg.norm(v)
    v = (1 - (2 * np.pi / norm) * np.ceil((norm - np.pi) / (2 * np.pi))) * v

    return v


def deparameterize_homog(v):
    # Given a parameterized homogeneous vector return its deparameterization
    """your code here"""
    norm = np.linalg.norm(v)

    a = np.cos(norm / 2)
    b = (sinc(norm / 2) / 2) * v

    a = np.array([[a]])
    b = b.reshape(-1,1)
    v_bar = np.vstack((a, b))

    return v_bar

def deparameterize_homog_with_Jacobian(v):
    # Given a parameterized homogeneous vector return its deparameterization and the Jacobian w.r.t parameters
    # Input:
    #    v - homogeneous parameterization vector (11x1 in case of p)
    # Output:
    #    v_bar - deparameterized homogeneous vector
    #    partial_vbar_partial_v - derivative of v_bar w.r.t v


    partial_vbar_partial_v = np.zeros((12,11))
    v_bar = np.zeros((12,1))

    """your code here"""
    norm = np.linalg.norm(v)
    
    a = np.cos(norm / 2)
    b = (sinc(norm / 2) / 2) * v

    if norm == 0:
        partial_a = np.zeros((1,11))
        partial_b = 0.5 * np.eye(11)
    else:
        partial_a = -0.5 * b.T
        partial_b = (sinc(norm/2) * 0.5) * np.eye(11) + ((1/(4*norm)) * d_sinc(norm/2) * (v @ v.T))

    a = np.array([[a]])
    b = b.reshape(11,1)
    v_bar = np.vstack((a, b))

    partial_vbar_partial_v[0] = partial_a
    partial_vbar_partial_v[1:] = partial_b

    return v_bar, partial_vbar_partial_v

def data_normalize_with_cov(pts, covarx):
    # Data normalization of n dimensional pts
    #
    # Input:
    #    pts - is in inhomogeneous coordinates
    #    covarx - covariance matrix associated with x. Has size 2n x 2n, where n is number of points.
    # Outputs:
    #    pts - data normalized points
    #    T - corresponding transformation matrix
    #    covarx - normalized covariance matrix

    """your code here"""
    s = np.sqrt(2 / np.sum(np.var(pts, axis=1)))
    mu = np.mean(pts, axis=1)

    T = np.eye(pts.shape[0]+1)
    T[0:2,0:2] *= s
    T[0:2,2] = -s * mu

    pts = homogenize(pts)
    pts = T @ pts

    J = np.eye(covarx.shape[0]) * s
    covarx = J @ covarx @ J.T

    return pts, T, covarx

def compute_cost(P, x, X, covarx):
    # Computes the total reprojection error
    # Inputs:
    #    P - the camera projection matrix
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    covarx - covariance matrix associated with x. Has size 2n x 2n, where n is number of points.
    # Output:
    #    cost - Total reprojection error

    """your code here"""
    X = homogenize(X)
    x_h = dehomogenize(P @ X)

    epsilon = (x - x_h).reshape(-1,1)
    sigma_i = np.linalg.inv(covarx)
    
    cost = epsilon.T @ sigma_i @ epsilon
    cost = cost[0,0]

    return cost

# %%
#Unit Tests (Do not change)

# partial_x_partial_p unit test
def check_values_partial_x_partial_p():
    eps = 1e-8  # Floating point error threshold
    x_2d = np.load('Unit_test/x_2d.npy')
    P = np.load('Unit_test/Projection.npy')
    X = np.load('Unit_test/X.npy')
    target = np.load('Unit_test/dx_dp.npy')
    dx_dp = partial_x_partial_p(P,X,x_2d)
    valid = np.all((dx_dp < target + eps) & (dx_dp > target - eps))
    print(f'Computed partial_x_partial_p is all equal to ground truth +/- {eps}: {valid}')

# deparameterize_homog_with_Jacobian unit test
def check_values_partial_vbar_partial_v():
    eps = 1e-8  # Floating point error threshold
    p = np.load('Unit_test/p.npy')
    dp_dp_target = np.load('Unit_test/dp_dp.npy')
    p_bar_target = np.load('Unit_test/Projection.npy').reshape(12,1)
    p_bar,dp_dp = deparameterize_homog_with_Jacobian(p)
    valid_dp_dp = np.all((dp_dp < dp_dp_target + eps) & (dp_dp > dp_dp_target - eps))
    valid_p_bar = np.all((p_bar < p_bar_target + eps) & (p_bar > p_bar_target - eps))
    valid = valid_dp_dp & valid_p_bar
    print(f'Computed v_bar,partial_vbar_partial_v is all equal to ground truth +/- {eps}: {valid}')

check_values_partial_x_partial_p()
check_values_partial_vbar_partial_v()

# %%
def estimate_camera_projection_matrix_nonlinear(P, x, X, max_iters, lam):
    # Levenberg Marquardt algorithm to estimate camera projection matrix
    # Input:
    #    P - initial estimate of P
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    max_iters - maximum number of iterations
    #    lam - lambda parameter
    # Output:
    #    P - Final P (3x4) obtained after convergence

    # data normalization
    covarx = np.eye(2*X.shape[1])
    x, T, covarx = data_normalize_with_cov(x, covarx)
    X, U = data_normalize(X)


    """your code here"""
    tau_1 = 1e-7
    tau_2 = 0

    P = T @ P @ np.linalg.inv(U)
    v = parameterize_matrix(P)

    x_t = dehomogenize(x)
    X_t = dehomogenize(X)

    # you may modify this so long as the cost is computed
    # at each iteration
    for i in range(max_iters):
        #compute previous: v, P, and cost
        v_bar, J_deparam = deparameterize_homog_with_Jacobian(v)
        P = v_bar.reshape(3, 4)
        cost_prev = compute_cost(P, x_t, X_t, covarx)

        #compute Jacobian
        _, n = X_t.shape
        J = np.zeros((2 * n, 12))

        x_proj = dehomogenize(P @ X)
        for k in range(n):
            X_k_hom = X[:, k]
            x_k = x_proj[:, k]
            J[2*k:2*k+2, :] = partial_x_partial_p(P, X_k_hom, x_k)

        J = J @ J_deparam        

        #compute normal equations matrix
        epsilon = (x_t - x_proj).flatten('F').reshape(-1,1)
        sigma_inv = np.linalg.inv(covarx)
        A = J.T @ sigma_inv @ J
        b = J.T @ sigma_inv @ epsilon

        while True:
            #compute candidate/current: v_0, P_0, cost_0
            delta = np.linalg.solve(A + lam * np.eye(11), b)
            v_0 = v + delta
            P_0 = deparameterize_matrix(v_0, 3, 4)
            cost_0 = compute_cost(P_0, x_t, X_t, covarx)

            #compare candidate to previous
            if cost_0 >= cost_prev:
                lam *= 10
            else:
                v = v_0
                P = P_0
                lam /= 10                
                break
        
        #termination criteria
        if (1 - (cost_0 / cost_prev)) <= tau_1:
            break
        if (cost_prev - cost_0) <= tau_2:
            break
        
        cost = compute_cost(P, x_t, X_t, covarx)
        print ('iter %03d Cost %.9f'%(i+1, cost))

    # data denormalization
    P = np.linalg.inv(T) @ P @ U
    return P



# LM hyperparameters
lam = .001
max_iters = 100

# Run LM initialized by DLT estimate with data normalization
print ('Running LM with data normalization')
print ('iter %03d Cost %.9f'%(0, cost))
time_start=time.time()
P_LM = estimate_camera_projection_matrix_nonlinear(P_DLT, x, X, max_iters, lam)
time_total=time.time()-time_start
print('took %f secs'%time_total)

print("\n==Correct outputs==")
print("Begins at %.9f; ends at %.9f"%(84.104680130, 82.790238005))

# %%
# Report your P_LM (estimated camera projection matrix nonlinear) final value here!
display_results(P_LM, x, X, 'P_LM')


