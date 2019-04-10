{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def initialize_spin_lattice(N):\n",
    "    #just combined the creation of the lattice with the relationship between each electron and its\n",
    "    #neighbors. It actually cut down on hte run time. Found something like this in the same quantum textbook mentioned further down, \n",
    "    initial_spins_la = -pd.DataFrame(np.ones((N, N)))\n",
    "    \n",
    "    initial_spins_melted = pd.melt(initial_spins_la.reset_index(level=0),id_vars=[\"index\"])\n",
    "    initial_spins_melted.columns = [\"X\",\"Y\",\"spin\"]\n",
    "    initial_spins_melted.reset_index(inplace=True)\n",
    "    initial_spins_melted.set_index([\"X\", \"Y\"],inplace=True)\n",
    "    \n",
    "    initial_spins_melted[\"right_spin\"] = initial_spins_melted.groupby('Y')[\"index\"].shift(-1)\n",
    "    initial_spins_melted[\"left_spin\"] = initial_spins_melted.groupby('Y')[\"index\"].shift(1)\n",
    "    initial_spins_melted[\"up_spin\"] = initial_spins_melted.groupby('X')[\"index\"].shift(1)\n",
    "    initial_spins_melted[\"down_spin\"] = initial_spins_melted.groupby('X')[\"index\"].shift(-1)\n",
    "    \n",
    "    initial_spins_melted.sort_index(inplace=True)\n",
    "    \n",
    "    initial_spins_melted.loc[(0),\"left_spin\"] = np.array(initial_spins_melted[\"index\"][N-1])\n",
    "    initial_spins_melted.loc[(N-1),\"right_spin\"] = np.array(initial_spins_melted[\"index\"][0])\n",
    "    initial_spins_melted.loc[(slice(None),0),\"up_spin\"]  = np.array(initial_spins_melted[\"index\"][:,N-1])\n",
    "    initial_spins_melted.loc[(slice(None),N-1),\"down_spin\"] = np.array(initial_spins_melted[\"index\"][:,0])\n",
    "    \n",
    "    initial_spins_melted[\"right_spin\"]= initial_spins_melted[\"right_spin\"].astype(int)\n",
    "    initial_spins_melted[\"left_spin\"] = initial_spins_melted[\"left_spin\"].astype(int)\n",
    "    initial_spins_melted[\"up_spin\"]   = initial_spins_melted[\"up_spin\"].astype(int)\n",
    "    initial_spins_melted[\"down_spin\"] = initial_spins_melted[\"down_spin\"].astype(int)\n",
    "\n",
    "    initial_spins_melted.reset_index(inplace=True)\n",
    "    initial_spins_melted.set_index(\"index\", inplace=True)\n",
    "    \n",
    "    return initial_spins_melted"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def deltaE(s,sum_j,h,J):#Used to determine if the spins will flip or not\n",
    "    return 2*s*(J*sum_j + h)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "def flip(delta,T,k=1):#What does the flipping\n",
    "    delta = np.asarray(delta)\n",
    "    return np.where(np.exp(-delta/(k*T)) > np.random.uniform(),-1,1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "def one_step(la,T,parityX,parityY,h=0,frac=0.7,):\n",
    "    \n",
    "    sample_spins = la.loc[(la[\"X\"] % 2 == parityX) & (la[\"Y\"] % 2 == parityY)]#Took a week to figure this out.\n",
    "    #coudnt get the predicted answers until I found something similar in a code for a quantum machanics question from my old textbook.\n",
    "    sample_spins = sample_spins.sample(frac=frac)  \n",
    "    \n",
    "    #The neighbors of each the electron are considred in this function.\n",
    "    sum_j = (la.loc[sample_spins[\"up_spin\"], \"spin\"].values+\n",
    "            la.loc[sample_spins[\"down_spin\"], \"spin\"].values+\n",
    "            la.loc[sample_spins[\"left_spin\"], \"spin\"].values+\n",
    "            la.loc[sample_spins[\"right_spin\"], \"spin\"].values)\n",
    "\n",
    "    s_i = sample_spins[\"spin\"]\n",
    "\n",
    "    delta_E = deltaE(s_i,sum_j,h=h,J=1)\n",
    "    \n",
    "    #Where the flips actually happen\n",
    "    \n",
    "    flip_array = flip(delta_E,T)\n",
    "    \n",
    "    la.loc[sample_spins.index, \"spin\"] *= flip_array\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#The overarching code responsible for the lattice as a whole\n",
    "def get_to_equilibrium(la,T,h,frac=0.8,iteration=500):\n",
    "    for i in range(iteration): \n",
    "        parity = np.random.choice([0,1], size=2)\n",
    "        one_step(la, T, parity[0], parity[1], h, frac)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#getting the magnetization is easy, while the energy requires mroe work to calculate.\n",
    "def compute_energy(la):\n",
    "    tot = 0\n",
    "    for i in la.index:\n",
    "        \n",
    "        tot = tot + 0.5 * -1 * la[\"spin\"].loc[i]*(la[\"spin\"].loc[la[\"right_spin\"].loc[i]]+\n",
    "                                                        la[\"spin\"].loc[la[\"left_spin\"].loc[i]]+\n",
    "                                                        la[\"spin\"].loc[la[\"up_spin\"].loc[i]]+\n",
    "                                                        la[\"spin\"].loc[la[\"down_spin\"].loc[i]])\n",
    "    return tot/len(la)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#arrays need to be made for the magnetic field values, both ascending and descending\n",
    "h_forward = np.arange(-5, 5, 0.3)\n",
    "h_backward = np.arange(5, -5, -0.3)\n",
    "#Easy to calcula\n",
    "def compute_magnetization(x):\n",
    "    return x.mean()\n",
    "#The ferromagnet is constructed here.\n",
    "spin_lattice = initialize_spin_lattice(32)  \n",
    "\n",
    "# hold equilibirum spins for each h\n",
    "h_forward_la = pd.DataFrame(index=spin_lattice.index, columns=h_forward) \n",
    "\n",
    "\n",
    "for i in h_forward:                                         # iterate through the h values\n",
    "    get_to_equilibrium(la = spin_lattice, T = 3, h = i)     # get to equilibrium\n",
    "    \n",
    "    h_forward_[i] = spin_lattice[\"spin\"]\n",
    "    \n",
    "# store magnetization values\n",
    "hf_mag = pd.DataFrame(columns=[\"Magnetization\"],index=h_forward)         \n",
    "\n",
    "# compute magnetization for each h value\n",
    "hf_mag[\"Magnetization\"] = h_forward_la.apply(compute_magnetization)\n",
    "## initialize spin lattice\n",
    "spin_lattice = initialize_spin_lattice(32)    \n",
    "\n",
    "# create hold equilibirum spins for each h\n",
    "h_backward_la = pd.DataFrame(index=spin_lattice.index, columns=h_backward)\n",
    "\n",
    "for i in h_backward:                                       \n",
    "    get_to_equilibrium(la = spin_lattice, T = 3, h = i)    \n",
    "    \n",
    "    h_backward_la[i] = spin_lattice[\"spin\"]\n",
    "    \n",
    "#store magnetization values   \n",
    "hb_mag = pd.DataFrame(columns=[\"Magnetization\"],index=h_backward)           \n",
    "\n",
    "## compute magnetization for each h value\n",
    "hb_mag[\"Magnetization\"] = h_backward_la.apply(compute_magnetization)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.plot(hb_mag.index,hb_mag[\"Magnetization\"],label=\"backward\")\n",
    "plt.plot(hf_mag.index,hf_mag[\"Magnetization\"],label=\"forward\")\n",
    "plt.xlabel(\"External Magnetic Field $h$\")\n",
    "plt.ylabel(\"Magnetization\")\n",
    "plt.axvline(linestyle='--',alpha=0.1)\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.6.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
