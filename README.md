
 "cells": [
  {
   "cell_type": "markdown",
   "id": "8a21fe2c",
   "metadata": {},
   "source": [
    "## AXIAL BEAM PROBLEM"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cacad1d6",
   "metadata": {},
   "source": [
    "<p>An axial beam problem involves analyzing a beam under axial load, where the beam deforms while its cross sections remain plane</p>\n",
    "<img src = \"beam.png\" style=\"width:250px;height:250px;\"/>\n",
    "<p>Above shown is an image of beam under axial load let say 10N, The idea is to solve this problem using the concept of direct fem so we will discreatize the beam into 10 equal elments this will result in 11 nodes to solve since the beam is a 1D body we will have only one variable at each node to solve </p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6a0bce44",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 720x72 with 10 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Import matplotlib.pyplot module\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Create a list of 10 elements\n",
    "elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n",
    "\n",
    "# Create a figure and a set of subplots\n",
    "fig, axes = plt.subplots(1, 10, figsize=(10, 1))\n",
    "\n",
    "# Loop through the subplots and plot the elements\n",
    "for i, ax in enumerate(axes.flat):\n",
    "    # Plot the element as a circle\n",
    "    ax.plot(elements[i], 0, 'o', markersize=20, color='blue')\n",
    "    # Add the element number as text\n",
    "    ax.text(elements[i], 0, str(elements[i]), fontsize=14, color='white', ha='center', va='center')\n",
    "    # Remove the axes ticks and labels\n",
    "    ax.set_xticks([])\n",
    "    ax.set_yticks([])\n",
    "\n",
    "# Adjust the spacing between subplots\n",
    "fig.tight_layout()\n",
    "\n",
    "# Show the figure\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "41516acd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "12fd2ea6",
   "metadata": {},
   "outputs": [],
   "source": [
    "columns = ['start','end','area','material']\n",
    "data = [[0,1,5,12],\n",
    "       [1,2,5,12],\n",
    "       [2,3,5,12],\n",
    "       [3,4,5,12],\n",
    "       [4,5,5,12],\n",
    "       [5,6,5,12],\n",
    "       [6,7,5,12],\n",
    "       [7,8,5,12],\n",
    "       [8,9,5,12],\n",
    "       [9,10,5,12],]\n",
    "elements = pd.DataFrame(data=data,columns=columns)\n",
    "col = ['coordinates','displacement','load']\n",
    "dat = [[0,0,0],\n",
    "       [5,np.nan,0],\n",
    "       [10,np.nan,0],\n",
    "       [15,np.nan,0],\n",
    "       [20,np.nan,0],\n",
    "       [25,np.nan,0],\n",
    "       [30,np.nan,0],\n",
    "       [35,np.nan,0],\n",
    "       [40,np.nan,0],\n",
    "       [45,np.nan,0],\n",
    "       [50,np.nan,10]]\n",
    "nodes = pd.DataFrame(data=dat,columns=col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "983ac4cf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>start</th>\n",
       "      <th>end</th>\n",
       "      <th>area</th>\n",
       "      <th>material</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>7</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8</td>\n",
       "      <td>9</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   start  end  area  material\n",
       "0      0    1     5        12\n",
       "1      1    2     5        12\n",
       "2      2    3     5        12\n",
       "3      3    4     5        12\n",
       "4      4    5     5        12\n",
       "5      5    6     5        12\n",
       "6      6    7     5        12\n",
       "7      7    8     5        12\n",
       "8      8    9     5        12\n",
       "9      9   10     5        12"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "elements"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "7db62404",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>coordinates</th>\n",
       "      <th>displacement</th>\n",
       "      <th>load</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>25</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>30</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>35</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>40</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>45</td>\n",
       "      <td>NaN</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>50</td>\n",
       "      <td>NaN</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    coordinates  displacement  load\n",
       "0             0           0.0     0\n",
       "1             5           NaN     0\n",
       "2            10           NaN     0\n",
       "3            15           NaN     0\n",
       "4            20           NaN     0\n",
       "5            25           NaN     0\n",
       "6            30           NaN     0\n",
       "7            35           NaN     0\n",
       "8            40           NaN     0\n",
       "9            45           NaN     0\n",
       "10           50           NaN    10"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nodes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5e7a0d2b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>start</th>\n",
       "      <th>end</th>\n",
       "      <th>area</th>\n",
       "      <th>material</th>\n",
       "      <th>length</th>\n",
       "      <th>stiffness</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>6</td>\n",
       "      <td>7</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>7</td>\n",
       "      <td>8</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>8</td>\n",
       "      <td>9</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>9</td>\n",
       "      <td>10</td>\n",
       "      <td>5</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>12.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   start  end  area  material  length  stiffness\n",
       "0      0    1     5        12       5       12.0\n",
       "1      1    2     5        12       5       12.0\n",
       "2      2    3     5        12       5       12.0\n",
       "3      3    4     5        12       5       12.0\n",
       "4      4    5     5        12       5       12.0\n",
       "5      5    6     5        12       5       12.0\n",
       "6      6    7     5        12       5       12.0\n",
       "7      7    8     5        12       5       12.0\n",
       "8      8    9     5        12       5       12.0\n",
       "9      9   10     5        12       5       12.0"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "def compute_length(e):\n",
    "    start_node = e['start']\n",
    "    end_node = e['end']\n",
    "    coordinate_start = nodes.loc[start_node,'coordinates']\n",
    "    coordinate_end = nodes.loc[end_node,'coordinates']\n",
    "    length = abs(coordinate_end-coordinate_start)\n",
    "    return length\n",
    "def compute_stiffnes(e):\n",
    "    stiffness = e['material']*e['area']/e['length']\n",
    "    return stiffness\n",
    "elements['length']=elements.apply(compute_length,axis=1)\n",
    "elements['stiffness']=elements.apply(compute_stiffnes,axis=1)\n",
    "elements"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "463974e6",
   "metadata": {},
   "source": [
    "<p>Till now we have discretized the beam into 10 elements now it is time to apply direct fem technique on one element lets say the node envolved be (i,j) </p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "dec17b40",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_force(e):\n",
    "    nodal_force = e['stiffness']*nodes['displacement']\n",
    "    return nodal_force"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78f5553e",
   "metadata": {},
   "source": [
    "<p> Now we need to generate global equation for each elements and sum up to get the final global matrix, we have our variable displacements at each nodes, and we will have our global equation after this step and hence we can invert the global matrix with load to get the nodal displacements at each node, lets do it</p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "d9fd5062",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compute_global_equation(e):\n",
    "#     e = elements.loc[0]\n",
    "    N = len(nodes)\n",
    "    k = np.zeros((N,N))\n",
    "    stiff = e['stiffness']\n",
    "    start = e['start'].astype(int)\n",
    "    end = e['end'].astype(int)\n",
    "    idx = np.ix_([start,end],[start,end])\n",
    "    k[idx] = stiff*np.array([[1,-1],[-1,1]])\n",
    "    return k\n",
    "k = elements.apply(compute_global_equation,axis=1)\n",
    "global_matrix = k.sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "558c19e8",
   "metadata": {},
   "source": [
    "<p> Here we have our global equation now we will compute our displacement at each nodes </p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "dc6c1592",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 12., -12.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [-12.,  24., -12.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0., -12.,  24., -12.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0., -12.,  24., -12.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0., -12.,  24., -12.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0., -12.,  24., -12.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0., -12.,  24., -12.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0., -12.,  24., -12.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0.,   0., -12.,  24., -12.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., -12.,  24., -12.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., -12.,  12.]])"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "global_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aaf1d052",
   "metadata": {},
   "source": [
    "<p>now is the time to apply boundry conditions and manipulate the global matrix according to the boundry condition.</p>\n",
    "<p>since the nodal displacement at node 1 will be zero so we can convert the global matrix in this form to satisfy our condition </p> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "fdeb33da",
   "metadata": {},
   "outputs": [],
   "source": [
    "#for the boundry condititon to satify\n",
    "global_matrix[0][1] = 0 "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "02b2bd06",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 12.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [-12.,  24., -12.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0., -12.,  24., -12.,   0.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0., -12.,  24., -12.,   0.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0., -12.,  24., -12.,   0.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0., -12.,  24., -12.,   0.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0., -12.,  24., -12.,   0.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0., -12.,  24., -12.,   0.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0.,   0., -12.,  24., -12.,   0.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., -12.,  24., -12.],\n",
       "       [  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., -12.,  12.]])"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "global_matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a86494d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "nodes['displacement'] = np.dot(np.linalg.inv(global_matrix),nodes['load'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "844a3360",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>coordinates</th>\n",
       "      <th>displacement</th>\n",
       "      <th>load</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>5</td>\n",
       "      <td>0.833333</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>10</td>\n",
       "      <td>1.666667</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15</td>\n",
       "      <td>2.500000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>20</td>\n",
       "      <td>3.333333</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>25</td>\n",
       "      <td>4.166667</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>30</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>35</td>\n",
       "      <td>5.833333</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>40</td>\n",
       "      <td>6.666667</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>45</td>\n",
       "      <td>7.500000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>50</td>\n",
       "      <td>8.333333</td>\n",
       "      <td>10</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    coordinates  displacement  load\n",
       "0             0      0.000000     0\n",
       "1             5      0.833333     0\n",
       "2            10      1.666667     0\n",
       "3            15      2.500000     0\n",
       "4            20      3.333333     0\n",
       "5            25      4.166667     0\n",
       "6            30      5.000000     0\n",
       "7            35      5.833333     0\n",
       "8            40      6.666667     0\n",
       "9            45      7.500000     0\n",
       "10           50      8.333333    10"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nodes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "da98546d",
   "metadata": {},
   "source": [
    "### post processing "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "dcc3d8e2",
   "metadata": {},
   "source": [
    "<p><b>Now since the nodal displacement is known we can calculate the strain and stress </b></p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b48da6f1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.16666666666666666"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### theoretically the strain should be F/(AE) lets verify\n",
    "theoretical_strain = 10/(5*12)\n",
    "theoretical_strain"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "cb36f710",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "strain at element 1 =  0.16666666666666646\n",
      "strain at element 2 =  0.16666666666666646\n",
      "strain at element 3 =  0.16666666666666652\n",
      "strain at element 4 =  0.16666666666666652\n",
      "strain at element 5 =  0.16666666666666635\n",
      "strain at element 6 =  0.1666666666666666\n",
      "strain at element 7 =  0.16666666666666644\n",
      "strain at element 8 =  0.1666666666666668\n",
      "strain at element 9 =  0.16666666666666644\n",
      "strain at element 10 =  0.16666666666666644\n"
     ]
    }
   ],
   "source": [
    "### calculated strain at each element\n",
    "N = len(nodes)\n",
    "for i in range(N-1):\n",
    "    strain = (nodes['displacement'][i+1]-nodes['displacement'][i])/5\n",
    "    print(f\"strain at element {i+1} = \",strain)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "872b8524",
   "metadata": {},
   "source": [
    "<p> as we can observe the calculated values are closer to the theoretical values of strain for the taken problem. </p>\n",
    "<p> Now lets verify for stress too </p>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "fcef3e52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2.0"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "### theoretically stress is F/A\n",
    "theretical_stress = 10/5\n",
    "theretical_stress"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "08a20134",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "stress at element 1 =  1.9999999999999976\n",
      "stress at element 2 =  1.9999999999999976\n",
      "stress at element 3 =  1.9999999999999982\n",
      "stress at element 4 =  1.9999999999999982\n",
      "stress at element 5 =  1.9999999999999962\n",
      "stress at element 6 =  1.9999999999999991\n",
      "stress at element 7 =  1.9999999999999973\n",
      "stress at element 8 =  2.0000000000000018\n",
      "stress at element 9 =  1.9999999999999973\n",
      "stress at element 10 =  1.9999999999999973\n"
     ]
    }
   ],
   "source": [
    "### calculated stress can be verified using the property stress = young's modulus * strain\n",
    "for i in range(N-1):\n",
    "    strain = (nodes['displacement'][i+1]-nodes['displacement'][i])/5\n",
    "    print(f\"stress at element {i+1} = \",strain*12) ### 12 is young's modulus in Mpa for this problem statement"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
