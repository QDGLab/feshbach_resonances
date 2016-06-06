#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import Tkinter as Tk
import os
import sys
import platform
import linecache
from fractions import Fraction
from sympy.physics.quantum.cg import CG

#some constants and unit conversions:
GAUSS_TO_TESLA = 10000
MHZ_TO_WAVENUMBERS = 29979.0
WAVENUMBERS_TO_HARTREE = 219474.63
BOHR_MAGNETON = 0.0000021271937						#in hartree/tesla
NUCLEAR_MAGNETON = 0.0000021271937/1836.152667 		#in hartree/tesla

def main():
	"""
	This code computes the energies of the bound states and the threshold energy within the asymptotic bound state model (ABM) over a 
	given range of the external magnetic field. It plots these energies as a function of the magnetic field, and identifies crossing 
	points and tangent points between the bound states and the threshold energy.
	The user can modify/save the plot with the matplotlib interface or with command lines (interactive mode is activated).
	
	All the physical parameters are stored and computed within the class Simulation. The object my_simulation is treated as a global
	variable.
	The classes InterfacePath and InterfaceData provide a graphical user interface (GUI). The first one sets a work directory and an
	input file for the data (optional), and the second one allows manual data input and modification of the input file (if provided).	
	"""
	plt.ion()									#Activate interactive mode
	global the_main_path
	if platform.system()=='Linux':
		the_main_path = r'/home/corneliu/Dropbox/Stage/abm_wrap/mode1/new'
	else:
		the_main_path = r'C:\Users\Corneliu\Dropbox\Stage\abm_wrap\mode1\new'
	global my_simulation
	my_simulation = Simulation()
	
	root_path = Tk.Tk(className = " Setting Work Directory")
	my_interfacePath = InterfacePath(root_path,my_simulation)
	
	root_data = Tk.Tk(className = " Setting Simulation Parameters")
	my_interfacePath = InterfaceData(root_data,my_simulation)
	
	my_simulation.update_basis()				#Defines the relevant basis for asymptotic and collision states
	my_simulation.update_H_rel()                #Computes the relative motion Hamiltonian
	my_simulation.update_H_hf()                 #Computes the hyperfine Hamiltonian
	my_simulation.find_hf_states()
	my_simulation.get_states()                	#Computes the energies of the bound/threshold states by diagonalizing the Hamiltonians
	my_simulation.reconvert_units()             #Back to atomic physics units
	my_simulation.get_crossings()               #Finds crossing/tangent points between the bound states and the threshold energy 
	
	root_plot = Tk.Tk(className = " Plot Parameters")			
	my_interfacePath = InterfacePlot(root_plot,my_simulation)

class Simulation(object):
	"""
	This class stores all the physical parameters and provides methods for computing the bound/threshold energies and for plotting.
	"""

	def __init__(self):
		"""
		Initializes the physical parameters at zero. The direct inputs from the user are listed bellow. The purely computational 
		attributes are explained in the code with comments. 
		
		User input attributes (the # stands for atom number # when some constants are defined for atom 1 and 2):
			path 				The path to a work directory containing the input file (if provided) and used for saving files
			input_file          A file containing the input parameters listed bellow. If not provided, manual input is possible
			hf_splitting#       Hyperfine Splittings of atom # (in MHz)
			gf#                 G-factor of atom #
			mu#                 Nuclear magnetic moment of atom #
			I#                  Nuclear spin of atom #
			Es                  Energy of the singlet least bound state (in cm^-1)
			Et                  Energy of the triplet least bound state (in cm^-1)
			ml                  Orbital projection
			mf#                 Hyerfine state of atom #
			B_min               Minimum of the magnetic field range
			B_max               Maximum of the magnetic field range
			B_step              Step of the magnetic field range
		"""
		self.path = the_main_path										#os.path.abspath(".")		#Initialized at the current location
		self.input_file = r'input_mode_1.txt'									#""
		self.hf_splitting1 = 0
		self.hf_splitting2 = 0
		self.gf1 = 0
		self.gf2 = 0
		self.gamma1 = 0							#Hyperfine constant of atom 1. Computed as hf_splitting1 * gf1
		self.gamma2 = 0							#Hyperfine constant of atom 2. Computed as hf_splitting2 * gf2
		self.mu1 = 0
		self.mu2 = 0
		self.I1 = 0
		self.I2 = 0
		self.Es = 0
		self.Et = 0
		self.ml = 0
		self.mf1 = 0
		self.mf2 = 0
		self.B_min = 0
		self.B_max = 0
		self.B_step = 0
		self.B_grid = []						#Values of the magnetic field: range [B_min , B_max] with step B_step
		self.basis_ass = np.array([])			#Uncoupled basis (basis [mi1,ms1,mi2,ms2]) for the asymptotic states
		self.basis_col = np.array([])			#Uncoupled basis (basis [mi1,ms1,mi2,ms2]) for the collision states
		self.num_ass = 0						#Number of threshold states
		self.num_col = 0						#Number of bound states
		self.H_rel = np.array([])				#Matrix of the relative motion Hamiltonian
		self.H_hf_ass = np.array([])			#Matrix of the hyperfine Hamiltonian on the asymptotic basis
		self.H_hf_col = np.array([])			#Matrix of the hyperfine Hamiltonian on the collision basis
		self.hf_states_ass = []
		self.E_threshold=np.array([])			#Values of the energy of each threshold state as a function of the magnetic field
		self.E_bound=np.array([])				#Values of the energy of each bound state as a function of the magnetic field
		self.singlet_part_ass = np.array([])	#Projection on Singlet subspace for each threshold state as a function of the magnetic field
		self.singlet_part_col = np.array([])	#Projection on Singlet subspace for each bound state as a function of the magnetic field
		self.crossings=np.array([])				#List of tangent points between threshold energy and bound states (B,E)
		self.tangents=np.array([])				#List of tangent points between threshold energy and bound states (B,E)
		
	def read_file(self):
		"""
		Reads the input file and initializes the input values.
		
		Note:
			Here, the values are initialized as strings so that the class InterfaceData can handle fractions properly
		
		Raises:
			sys.exit   if any error occurs while reading the file
		"""
		try:
			self.hf_splitting1 = linecache.getline(self.input_file, 5).split()[-1]
			self.hf_splitting2 = linecache.getline(self.input_file, 6).split()[-1]
			
			self.gf1 = linecache.getline(self.input_file, 9).split()[-1]
			self.gf2 = linecache.getline(self.input_file, 10).split()[-1]

			self.mu1 = linecache.getline(self.input_file, 13).split()[-1]
			self.mu2 = linecache.getline(self.input_file, 14).split()[-1]

			self.I1 = linecache.getline(self.input_file, 17).split()[-1]
			self.I2 = linecache.getline(self.input_file, 18).split()[-1]

			self.Es = linecache.getline(self.input_file, 21).split()[-1]
			self.Et = linecache.getline(self.input_file, 22).split()[-1]
			
			self.ml = linecache.getline(self.input_file, 29).split()[-1]

			self.mf1 = linecache.getline(self.input_file, 32).split()[-1]
			self.mf2 = linecache.getline(self.input_file, 33).split()[-1]

			self.B_min  = linecache.getline(self.input_file, 36).split()[-1]
			self.B_max  = linecache.getline(self.input_file, 37).split()[-1]
			self.B_step = linecache.getline(self.input_file, 38).split()[-1]
		except:
			sys.exit("It looks like there is something wrong with your input file... You can: \n" + \
			"   - Check the format of your input file \n" +\
			"   - Retry without an input file (you will be able to build a new one) \n\n" +\
			"Your input file was:  " + self.input_file + "\n\n")

	def update_basis(self):
		"""
		Constructs the asymptotic and collision basis. 
		Defines the number of channels as the number of states in the colision basis.
		
		A state [mi1,ms1,mi2,ms2] belongs to the asymptotic basis if:
			mi1 + ms1 = mf1  AND  mi2 + ms2 = mf2
			
		A state [mi1,mi2,ms1,ms2] belongs to the asymptotic basis if:
			mi1 + ms1 + mi2 + ms2 = mf1 + mf2
		"""
		Basis_ass_temp = []
		Basis_col_temp = []
		for ms1 in [-.5,.5]:
			for ms2 in [-.5,.5]:
				if abs(self.mf1-ms1)<=self.I1 and abs(self.mf2-ms2)<=self.I2:
					mi1 = self.mf1-ms1
					mi2 = self.mf2-ms2
					Basis_ass_temp+=[[mi1,ms1,mi2,ms2]]
					
				ms = ms1+ms2
				if abs(self.mf1+self.mf2-ms)<=self.I1+self.I2:
					for mi2 in np.arange(max(self.mf1+self.mf2-ms-self.I1,-self.I2),min(self.I1+self.mf1+self.mf2-ms,self.I2)+1):
						mi1=self.mf1+self.mf2-ms-mi2
						Basis_col_temp+=[[mi1,ms1,mi2,ms2]]
						
		self.basis_ass = np.array(Basis_ass_temp).transpose()
		self.basis_col = np.array(Basis_col_temp).transpose()
		self.num_ass = len(self.basis_ass[0,:])
		self.num_col = len(self.basis_col[0,:])
		
	def update_H_rel(self):
		"""
		Constructs the relative motion Hamiltonian on the collision basis.
		
		Note:
			Only half of the Hamiltonian is explicitly computed. The other half is deduced thanks to hermitian symmetry.
		"""
		N = self.num_col
		CG_singlet=np.zeros((N,N))
		CG_triplet=np.zeros((N,N))
		S1 = .5
		S2 = .5
		for i in range (N):
			mi1 = self.basis_col[0, i]
			ms1 = self.basis_col[1, i]
			mi2 = self.basis_col[2, i]
			ms2 = self.basis_col[3, i]
			for j in range (i+1):
				mi1_prime = self.basis_col[0, j]
				ms1_prime = self.basis_col[1, j]
				mi2_prime = self.basis_col[2, j]
				ms2_prime = self.basis_col[3, j]
				
				if mi1 == mi1_prime and mi2 == mi2_prime:		#Clebsch-Gordan coefficiants computed with sympy
					CG_singlet[i,j] = float(CG(S1,ms1,S2,ms2,0,0).doit() * CG(S1,ms1_prime,S2,ms2_prime,0,0).doit())
					CG_triplet[i,j] = float(sum(CG(S1,ms1,S2,ms2,1,MS).doit() * CG(S1,ms1_prime,S2,ms2_prime,1,MS).doit() for MS in [-1,0,1]))
		CG_singlet -= np.diag(np.diag(CG_singlet)/2)			#divide the diagonal by 2 before symmetrization
		CG_triplet -= np.diag(np.diag(CG_triplet)/2)			#divide the diagonal by 2 before symmetrization
		
		self.H_rel = self.Es*(CG_singlet+CG_singlet.transpose()) + self.Et*(CG_triplet+CG_triplet.transpose())
					
	def update_H_hf(self):
		"""
		Initializes the Hyperfine Hamiltonian for asymptotic and collision basis
		"""
		self.H_hf_ass = self.calc_H_hf(self.basis_ass)
		self.H_hf_col = self.calc_H_hf(self.basis_col)
		
	def calc_H_hf(self,Basis):
		"""
		Computes the Hyperfine Hamiltonian for a given {[mi1,ms1,mi2,ms2]} basis.
		
		Args:
			Basis   The {[mi1,ms1,mi2,ms2]} basis over which the Hyperfine Hamiltonian is computed
		
		Returns:
			A N*N array with the Hyperfine Hamiltonian, with N the number of states in the basis
			
		Note:
			Only half of the Hamiltonian is explicitly computed. The other half is deduced thanks to hermitian symmetry.
		"""
		N = len(Basis[0,:])
		H=np.zeros((N,N))
		S1 = .5
		S2 = .5
		for i in range (N):
			mi1 = Basis[0, i]
			ms1 = Basis[1, i]
			mi2 = Basis[2, i]
			ms2 = Basis[3, i]
			H[i, i] = .5*(self.gamma1*ms1*mi1 + self.gamma2*ms2*mi2) 	#multiply the diagonal by .5 for future symmetrization
			for j in range (i):											#Computes the off-diagonal coefficients
				mi1_prime = Basis[0, j]
				ms1_prime = Basis[1, j]
				mi2_prime = Basis[2, j]
				ms2_prime = Basis[3, j]
				
				if mi2 == mi2_prime and ms2 == ms2_prime :
					if mi1 == mi1_prime-1 and ms1==ms1_prime+1:
						H[i, j] = np.sqrt(S1*(S1+1) - ms1_prime*(ms1_prime+1)) * np.sqrt(self.I1*(self.I1+1) - mi1_prime*(mi1_prime-1))*0.5*self.gamma1
					if mi1 == mi1_prime+1 and ms1==ms1_prime-1:
						H[i, j] = np.sqrt(S1*(S1+1) - ms1_prime*(ms1_prime-1)) * np.sqrt(self.I1*(self.I1+1) - mi1_prime*(mi1_prime+1))*0.5*self.gamma1
						
				if mi1 == mi1_prime and ms1 == ms1_prime :
					if mi2 == mi2_prime-1 and ms2==ms2_prime+1:
						H[i, j] = np.sqrt(S2*(S2+1) - ms2_prime*(ms2_prime+1)) * np.sqrt(self.I2*(self.I2+1) - mi2_prime*(mi2_prime-1))*0.5*self.gamma2
					if mi2 == mi2_prime+1 and ms2==ms2_prime-1:
						H[i, j] = np.sqrt(S2*(S2+1) - ms2_prime*(ms2_prime-1)) * np.sqrt(self.I2*(self.I2+1) - mi2_prime*(mi2_prime+1))*0.5*self.gamma2
						
		return H + H.transpose()
	
	def find_hf_states(self):
		self.hf_states_ass=np.zeros((self.num_ass,2))
		eigenvals_hf_ass, eigenvects_hf_ass = np.linalg.eigh(self.H_hf_ass)
		for F1 in [self.I1-.5,self.I1+.5]:
			for F2 in [self.I2-.5,self.I2+.5]:
				CG_vector = np.array([float((CG(self.I1,self.basis_ass[0,i],.5,self.basis_ass[1,i],F1,self.mf1) * \
					CG(self.I2,self.basis_ass[2,i],.5,self.basis_ass[3,i],F2,self.mf2)).doit()) for i in range(self.num_ass)])
				index = np.where(np.dot(eigenvects_hf_ass.transpose(), CG_vector)**2 >.9)[0]
				self.hf_states_ass[index,:]=[F1,F2]
	
	def get_states(self):
		"""
		Executes a loop over all values of B to compute the threshold and bound energies as a function of the magnetic field.
		
		Within the loop:
			Computes the Zeeman Hamiltonian for collision and asymptotic basis
			Computes the threshold and bound energies by diagonalizing thetotal asymptotic and collision Hamiltonians.
			Displays a progress bar over the main loop
		"""
		E_threshold_temp = []
		E_bound_temp = []
		singlet_part_ass_temp = []
		singlet_part_col_temp = []
		
		vector_CG_ass = np.array([float(CG(.5,self.basis_ass[1,i],.5,self.basis_ass[3,i],0,0).doit()) \
		for i in range (self.num_ass)])**2                                                                  #To compute the Singlet amplitude
		vector_CG_col = np.array([float(CG(.5,self.basis_col[1,i],.5,self.basis_col[3,i],0,0).doit()) \
		for i in range (self.num_col)])**2                                                                  #with matrices--> Increases speed
		
		if self.I1!=0 and self.I2!=0:			#This is the normal case
			vector_zeeman = np.array([-NUCLEAR_MAGNETON*self.mu1/self.I1, 2*BOHR_MAGNETON,\
			-NUCLEAR_MAGNETON*self.mu2/self.I2, 2*BOHR_MAGNETON])                               #To compute H_Zeeman with matrices
		
		elif self.I1==0 and self.I2!=0:			#Just in case of a zero nuclear spin..
			vector_zeeman = np.array([0, 2*BOHR_MAGNETON,\
			-NUCLEAR_MAGNETON*self.mu2/self.I2, 2*BOHR_MAGNETON])
				
		elif self.I1!=0 and self.I2==0:			#Just in case of a zero nuclear spin..
			vector_zeeman = np.array([-NUCLEAR_MAGNETON*self.mu1/self.I1, 2*BOHR_MAGNETON,\
			0, 2*BOHR_MAGNETON])
				
		elif self.I1==0 and self.I2==0:			#Just in case of a zero nuclear spin..		
			vector_zeeman = np.array([0, 2*BOHR_MAGNETON,\
			0, 2*BOHR_MAGNETON])
			
		matrix_zeeman_ass = np.diag(np.dot(self.basis_ass.transpose(), vector_zeeman))
		matrix_zeeman_col = np.diag(np.dot(self.basis_col.transpose(), vector_zeeman))

		i=0																		#Progress bar
		progress=0												                #Progress bar
		print "\n\n"                                                            #Progress bar
		sys.stdout.write("\rProgress of the main loop:  [" + "-"*40 + "]")      #Progress bar
		sys.stdout.flush()                                                      #Progress bar
		
		for B in self.B_grid:
			H_ze_ass = B*matrix_zeeman_ass
			H_ze_col = B*matrix_zeeman_col
			
			H_ass = self.H_hf_ass + H_ze_ass
			H_col = self.H_hf_col + H_ze_col + self.H_rel
			
			eigenvals_ass, eigenvects_ass = np.linalg.eigh(H_ass)
			eigenvals_col, eigenvects_col = np.linalg.eigh(H_col)
			
			singlet_part_ass_temp += [list(np.dot(eigenvects_ass.transpose()**2, vector_CG_ass))]
			singlet_part_col_temp += [list(np.dot(eigenvects_col.transpose()**2, vector_CG_col))]	

			E_threshold_temp += [eigenvals_ass]							#np.linalg.eigh returns the eigenvalues in ascending order
			E_bound_temp += [eigenvals_col]
			i+=1
			if int(i/len(self.B_grid)*40) > progress:                                                            #Progress bar
				progress = int(i/len(self.B_grid)*40)                                                            #Progress bar
				sys.stdout.write("\rProgress of the main loop:  [" + "#"*progress+"-"*(40-progress) + "]")       #Progress bar
				sys.stdout.flush()                                                                               #Progress bar
		sys.stdout.write("\rProgress of the main loop:  [" + "#"*40 + "]   :   DONE")                            #Progress bar
		sys.stdout.flush()                                                                                       #Progress bar
		print "\n\n"                                                                                             #Progress bar

		self.E_threshold=np.array(E_threshold_temp)
		self.E_bound=np.array(E_bound_temp)
		self.singlet_part_ass = np.array(singlet_part_ass_temp)
		self.singlet_part_col = np.array(singlet_part_col_temp)
		
	def calc_H_ze(self,Basis,Bvalue):
		"""
		Computes the Zeeman Hamiltonian for a given {[mi1,ms1,mi2,ms2]} basis.
		
		Args:
			Basis		The {[mi1,ms1,mi2,ms2]} basis over which the Hyperfine Hamiltonian is computed
			Bvalue 		The value of the magnetic field
		
		Returns:
			A N*N array with the Hyperfine Hamiltonian, with N the number of states in the basis
			
		Note:
			The Zeeman Hamiltonian is diagonal in the [mi1,ms1,mi2,ms2] representation
		"""	
		N = len(Basis[0,:])
		H=[]
		
		if self.I1!=0 and self.I2!=0:		#This is the normal case
			for i in range (N):
				mi1 = Basis[0, i]
				ms1 = Basis[1, i]
				mi2 = Basis[2, i]
				ms2 = Basis[3, i]
				H+=[2*BOHR_MAGNETON*Bvalue*(ms1+ms2) - Bvalue*NUCLEAR_MAGNETON*(mi1*self.mu1/self.I1 + mi2*self.mu2/self.I2)]
		
		elif self.I1==0 and self.I2!=0:		#Just in case of a zero nuclear spin..
			for i in range (N):
				mi1 = Basis[0, i]
				ms1 = Basis[1, i]
				mi2 = Basis[2, i]
				ms2 = Basis[3, i]
				H+=[2*BOHR_MAGNETON*Bvalue*(ms1+ms2) - Bvalue*NUCLEAR_MAGNETON*mi2*self.mu2/self.I2]
				
		elif self.I1!=0 and self.I2==0:		#Just in case of a zero nuclear spin..
			for i in range (N):
				mi1 = Basis[0, i]
				ms1 = Basis[1, i]
				mi2 = Basis[2, i]
				ms2 = Basis[3, i]
				H+=[2*BOHR_MAGNETON*Bvalue*(ms1+ms2) - Bvalue*NUCLEAR_MAGNETON*mi1*self.mu1/self.I1]
				
		elif self.I1==0 and self.I2==0:		#Just in case of a zero nuclear spin..
			for i in range (N):
				mi1 = Basis[0, i]
				ms1 = Basis[1, i]
				mi2 = Basis[2, i]
				ms2 = Basis[3, i]
				H+=[2*BOHR_MAGNETON*Bvalue*(ms1+ms2)]
			
		return np.diag(H)
		
	def reconvert_units(self):
		"""
		The computations were made in Hartree/Tesla. 
		Comes back to atomic physics units cm^-1/Gauss
		"""
		self.E_threshold *= WAVENUMBERS_TO_HARTREE
		self.E_bound *= WAVENUMBERS_TO_HARTREE
		self.B_grid *= GAUSS_TO_TESLA
	
	def get_crossings(self):
		"""
		Finds the crossing and tangent points between threshold energy and bound states
		"""
		N = self.num_col
		real_crossings = []
		tangent_crossings = []
		for jj in range(len(self.basis_ass[0,:])):
			for i in range(N):
				dE = self.E_bound[:,i]-self.E_threshold[:,jj]
				real_crossings_index = np.where(np.diff(np.sign(dE)))[0] #finds i so that dE[i]*dE[i+1]<0, i.e. dE changes signs after index i
				for k in real_crossings_index:		#Linear interpolation of the points before and after the crossing
					if k < len(self.B_grid)-1:		#Avoids IndexError
						B_points = np.array([self.B_grid[k],self.B_grid[k+1]])
						dE_points = np.array([dE[k],dE[k+1]])
						E_threshold_points = np.array([self.E_threshold[k,jj],self.E_threshold[k+1,jj]])
						interp_dE = np.poly1d(np.polyfit(B_points, dE_points, 1))
						interp_E_threshold = np.poly1d(np.polyfit(B_points, E_threshold_points, 1))
						B_cross = interp_dE.r[0]
						E_cross = interp_E_threshold (B_cross)
						real_crossings+=[[B_cross,E_cross]]
				
				#We have a tangent point if dE has a local extremum with value EXT = 0. For computational reasons, we don't look for extrema with 
				#value EXT = 0 (the function is discretized on the B range) but with values abs(EXT) < SOMETHING
				hypothetical_tangent_index = np.where(np.diff(np.sign(np.diff(dE))))[0] #the derivative of dE changes signs after index i --> extremum
				for k in hypothetical_tangent_index:	
					if k < len(self.B_grid)-2:			#Avoids IndexError
						if np.abs(dE[k+1])<= 2*np.max(np.abs(dE[k+1]-dE[k]),np.abs(dE[k+2]-dE[k+1])): #Our criterion for the SOMETHING we just discussed
							tangent_crossings+=[[self.B_grid[k+1],self.E_threshold[k+1,jj]]]
		self.crossings = np.array(real_crossings)
		self.tangents = np.array(tangent_crossings)
		
	def plot_result(self,fig_1, fig_2, keep_same_fig, bound_fig_1, list_threshold_fig_1, crossings_fig_1, \
		bound_fig_2, list_threshold_fig_2):
		"""
		Plots the threshold and bound states energies as a function of the magnetic field and displays the result. 
		
		Note:
			The interactive mode of matplotlib.pyplot is activated. Therefore, you can modify/save the plot either 
			with the graphic interface or directly in the terminal.
		"""
		if fig_2:
			if not keep_same_fig: plt.figure(2,figsize=(20,10))
			
			for i in range(self.num_ass):
				if list_threshold_fig_2 [i]:
					plt.plot(self.B_grid,self.singlet_part_ass[:,i],'--',linewidth = 3,label = 'State of Threshold {} : F1 = '.format(i+1) +\
						str(Fraction(self.hf_states_ass[i,0])) + ',  F2 = ' + str(Fraction(self.hf_states_ass[i,1])))		
			
			if bound_fig_2:
				colormap = plt.cm.gist_ncar
				plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, self.num_col)])
				
				for i in range(self.num_col):
					plt.plot(self.B_grid, self.singlet_part_col[:,i])	
					
			if not keep_same_fig:	
				plt.suptitle(r'Projection on the Singlet subspace for state $|$' + \
				str(Fraction(self.mf1)) + ', ' + str(Fraction(self.mf2))+r'$\rangle$', fontsize = 34)			
				
				plt.xlabel(r'$\mathrm{B}$ $\mathrm{[G]}$', fontsize = 28)
				plt.ylabel(r'$|\alpha|^2$  in  $|\Psi\rangle = \alpha|\Psi_S \rangle + \beta|\Psi_T \rangle$', fontsize = 28)
				plt.xticks(fontsize = 24)
				plt.yticks(fontsize = 24)
			plt.legend(loc='best',fontsize = 28)
		
		if fig_1:
			if not keep_same_fig: plt.figure(1,figsize=(20,10))

			for i in range(self.num_ass):
				if list_threshold_fig_1 [i]:
					plt.plot(self.B_grid,self.E_threshold[:,i],'--',linewidth = 3,label = 'Energy of Threshold {} : F1 = '.format(i+1) +\
						str(Fraction(self.hf_states_ass[i,0])) + ',  F2 = ' + str(Fraction(self.hf_states_ass[i,1])))
					
			if bound_fig_1:
				colormap = plt.cm.gist_ncar
				plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, self.num_col)])
				
				for i in range(self.num_col):
					plt.plot(self.B_grid, self.E_bound[:,i])
			if crossings_fig_1:
				if len(self.crossings)>0:	#Avoids IndexError
					plt.plot(self.crossings[:,0],self.crossings[:,1],'go',markersize=6,label = 'Crossings')# {}'.format(list(np.sort(self.crossings[:,0]))))
					
				if len(self.tangents)>0:	#Avoids IndexError
					plt.plot(self.tangents[:,0],self.tangents[:,1],'mo',markersize=6,label = 'Tangent points')
					
			if not keep_same_fig:		
				plt.suptitle(r'Expected resonnances for state $|$' + str(Fraction(self.mf1)) + ', ' + str(Fraction(self.mf2))+r'$\rangle$', fontsize = 34)	
				plt.xlabel(r'$\mathrm{B}$ $\mathrm{[G]}$', fontsize = 28)
				plt.ylabel(r'$\mathrm{E}$ $\mathrm{[cm^{-1}]}$', fontsize = 28)
				plt.xticks(fontsize = 24)
				plt.yticks(fontsize = 24)
			plt.legend(loc='best',fontsize = 28)		
		plt.show()
		
class InterfacePath(object):
	"""
	GUI interface built with the Tkinter module (Tk).
	Asks the user for a work directory and an optional input file.
	Can construct a work directory if necessary.
	
	Note:
		When main() is executed, this class modifies the attributes of the global variable 'my_simulation'.
	"""
	def __init__(self,root,simulation_object):
		"""
		Main Tk application.
		Asks the user for a work directory and an optional input file.
		
		Calls:
			check_input 		when pressing 'Done'
			user_close			if the user closes the main window (--> sys.exit)			
		"""
		self.root=root
		self.simulation=simulation_object
		self.frame = Tk.Frame(self.root, width=768, height=576, borderwidth=1)
		self.frame.pack()
		
		self.label_input_directory = Tk.Label(self.frame,justify=Tk.LEFT, text= \
		" Please enter the path to your work directory (e.g. '/home/user/abm'). \n You can enter '.' for the current directory.")
		self.label_input_directory.pack()
		
		self.var_directory = Tk.StringVar()
		self.var_directory.set(self.simulation.path)
		self.entry_directory = Tk.Entry(self.frame, textvariable=self.var_directory, width=150)
		self.entry_directory.pack()

		self.label_input_file = Tk.Label(self.frame, justify=Tk.LEFT, text= \
		" If you want to use an input file for constants, write its name here (e.g. 'input.txt'). \n This file must be in your work directory. \n For manual input, let empty.")
		self.label_input_file.pack()
		
		self.var_file = Tk.StringVar()
		self.var_file.set(self.simulation.input_file)
		self.entry_file = Tk.Entry(self.frame, textvariable=self.var_file, width=150)
		self.entry_file.pack()
		
		self.quit_button = Tk.Button(self.frame, text="Done", command = self.check_input)
		self.quit_button.pack()
		
		self.root.protocol("WM_DELETE_WINDOW", self.user_close)

		self.root.mainloop()
	
	def check_input(self):
		"""
		Checks if the directory and the input file given by the user exist.
		
		Calls:
			Simulation.read_file(self.simulation)	AND
			exit_interface									if a correct input file is given
			
			exit_interface									if no input file is given (--> manual input)
			
			warning_no_directory							if the work directory does not exist
			warning_no_file									if the input file does not exist
		"""
		dir_path = self.var_directory.get()
		file_path = self.var_file.get()
		self.simulation.path = os.path.normcase(dir_path)
		self.simulation.input_file = os.path.join(self.simulation.path,os.path.normcase(file_path))
		
		if not os.path.isdir(self.simulation.path):	
			self.warning_no_directory()
		
		elif file_path=='' or os.path.normpath(self.simulation.path)==os.path.normpath(self.simulation.input_file):
			self.exit_interface()
			
		elif os.path.isfile(self.simulation.input_file):
			self.exit_interface()
			self.simulation.read_file()
			
		else: 
			self.warning_no_file()
		
	def warning_no_directory(self):
		"""
		Is called if the work directory does not exist. 
		Displays a warning window allowing to construct the directory.
		
		Calls:
			build_dir		when pressing 'Build'
		
		Back to main interface when pressing 'Back'
		"""
		self.window_no_directory = Tk.Toplevel(self.root)
		message = Tk.Label(self.window_no_directory, justify=Tk.LEFT, text= \
		" The directory \n '" + self.simulation.path + \
		"' \n does not exist. \n Do you want to build it? \n (It might also be because of a blank at he end of your path..)")
		message.pack()
		back_button = Tk.Button(self.window_no_directory, text="Go Back", command = self.window_no_directory.destroy)
		back_button.pack(side="left")
		build_button = Tk.Button(self.window_no_directory, text="Build", command = self.build_dir)
		build_button.pack(side="right")
		
	def warning_no_file(self):
		"""
		Is called if the input file does not exist. 
		Displays a warning window askint for a new file or for manual input.
		
		Calls:
			exit_interface		when pressing 'Continue'
		
		Back to main interface when pressing 'Retry'
		"""
	
		self.window_no_file = Tk.Toplevel(self.root)
		message = Tk.Label(self.window_no_file, justify=Tk.LEFT, text= \
		" The file \n '" + self.simulation.input_file + "' \n was not found. \n Continue without input file?")
		message.pack()
		back_button = Tk.Button(self.window_no_file, text="Retry", command = self.window_no_file.destroy)
		back_button.pack(side="left")
		build_button = Tk.Button(self.window_no_file, text="Continue", command = self.exit_interface)
		build_button.pack(side="right")
		
	def build_dir(self):
		"""
		Builds a new work directory with the specified path and destroys the previous warning window.
		
		Calls:
			check_input
		"""
		os.mkdir(self.simulation.path)
		self.window_no_directory.destroy()
		self.check_input()
		
	def exit_interface(self):
		"""
		Normal exit. Alows the main program of the code to continue
		"""
		self.root.destroy()
		
	def user_close(self):
		"""
		Abnormal exit.
		
		Raises:
			sys.exit
		"""
		self.root.destroy()
		sys.exit("You closed the main application")
		
class InterfaceData(object):
	"""
	GUI interface built with the Tkinter module (Tk).
	Asks the user for the input parameters, and proceeds to basic tests. 
	The values are pre-filled with an input file if provided.
	Allows to save the new settings as a future input file.
	
	Note:
		When main() is executed, this class modifies the attributes of the global variable 'my_simulation'.
	"""
	
	
	def __init__(self,root,simulation_object):
		"""
		Main Tk application.
		Asks the user for the input parameters.
		
		Calls:
			get_data 			when pressing 'OK !'
			user_close			if the user closes the main window (--> sys.exit)
		"""
		self.root=root
		self.simulation=simulation_object
		self.main_left_frame = Tk.Frame(self.root, height = 350, width = 850, borderwidth=1)
		self.main_left_frame.pack(side=Tk.LEFT)
		
		self.main_right_frame = Tk.Frame(self.root, height = 350, width = 850, borderwidth=1)
		self.main_right_frame.pack(side=Tk.RIGHT)	
		
		self.build_frame_constants()
		self.build_frame_inputs()
		
		self.quit_button = Tk.Button(self.main_right_frame, text='OK !',fg='red', command = self.get_data)
		self.quit_button.pack(side=Tk.BOTTOM)
		
		self.root.protocol("WM_DELETE_WINDOW", self.user_close)
		
		self.root.mainloop()
		
	def build_frame_constants(self):
		"""
		Builds the frame asking for atomic constants
		"""
		self.frame_constants = Tk.Frame(self.main_left_frame, borderwidth=2, relief=Tk.GROOVE,height = 300, width = 800)
		self.frame_constants.pack(side=Tk.LEFT, padx=10, pady=10)
		
		self.label_constants = Tk.Label(self.frame_constants, text='ATOMIC CONSTANTS AND ENERGIES')
		self.label_constants.pack() 
		
		self.var_hf_splitting1 = Tk.StringVar()
		self.var_hf_splitting2 = Tk.StringVar()
		self.var_hf_splitting1.set(self.simulation.hf_splitting1)
		self.var_hf_splitting2.set(self.simulation.hf_splitting2)
		self.build_2subframe(self.frame_constants, 'Hyperfine Splittings (MHz)', 'Atom 1', 'Atom 2', self.var_hf_splitting1, self.var_hf_splitting2)
		
		self.var_g_factor1 = Tk.StringVar()
		self.var_g_factor2 = Tk.StringVar()
		self.var_g_factor1.set(self.simulation.gf1)
		self.var_g_factor2.set(self.simulation.gf2)
		self.build_2subframe(self.frame_constants, 'G-Factors', 'gF 1', 'gF 2', self.var_g_factor1, self.var_g_factor2)
		
		self.var_magn_moment1 = Tk.StringVar()
		self.var_magn_moment2 = Tk.StringVar()
		self.var_magn_moment1.set(self.simulation.mu1)
		self.var_magn_moment2.set(self.simulation.mu2)
		self.build_2subframe(self.frame_constants, 'Nuclear Magnetic Moments (Nuclear Bohr Magnetons)', 'mu 1', 'mu 2', self.var_magn_moment1, self.var_magn_moment2)
		
		self.var_nucl_spin1 = Tk.StringVar()
		self.var_nucl_spin2 = Tk.StringVar()
		self.var_nucl_spin1.set(self.simulation.I1)
		self.var_nucl_spin2.set(self.simulation.I2)
		self.build_2subframe(self.frame_constants, 'Nuclear Spins', 'I 1', 'I 2', self.var_nucl_spin1, self.var_nucl_spin2)		
		
		self.var_lbs_energy1 = Tk.StringVar()
		self.var_lbs_energy2 = Tk.StringVar()
		self.var_lbs_energy1.set(self.simulation.Es)
		self.var_lbs_energy2.set(self.simulation.Et)
		self.build_2subframe(self.frame_constants, 'Least Bound State Energy', 'Singlet', 'Triplet', self.var_lbs_energy1, self.var_lbs_energy2)
		
	def build_frame_inputs(self):
		"""
		Builds the frame asking for experimental conditions
		"""
		self.frame_inputs = Tk.Frame(self.main_right_frame, borderwidth=2, relief=Tk.GROOVE,height = 300, width = 800)
		self.frame_inputs.pack(side=Tk.TOP, padx=10, pady=10)
		
		self.label_inputs = Tk.Label(self.frame_inputs, text='INPUT STATE AND MAGNETIC FIELD RANGE')
		self.label_inputs.pack() 
		
		self.var_orbital_proj = Tk.StringVar()
		self.var_orbital_proj.set(self.simulation.ml)
		self.build_1subframe(self.frame_inputs, 'Orbital Projection', 'ML', self.var_orbital_proj)
		
		self.var_hf_state1 = Tk.StringVar()
		self.var_hf_state2 = Tk.StringVar()
		self.var_hf_state1.set(self.simulation.mf1)
		self.var_hf_state2.set(self.simulation.mf2)
		self.build_2subframe(self.frame_inputs, 'Hyperfine Input State', 'MF 1', 'MF 2', self.var_hf_state1, self.var_hf_state2,align=Tk.LEFT)
		
		self.var_B_min = Tk.StringVar()
		self.var_B_max = Tk.StringVar()
		self.var_B_step = Tk.StringVar()
		self.var_B_min.set(self.simulation.B_min)
		self.var_B_max.set(self.simulation.B_max)
		self.var_B_step.set(self.simulation.B_step)
		self.build_3subframe(self.frame_inputs, 'Magnetic Field Range and Step Size (Gauss)', 'Minimum Magnetic Field', \
		'Maximum Magnetic Field', 'Magnetic Field Step Size', self.var_B_min, self.var_B_max, self.var_B_step)
		
		
	def build_1subframe(self, master, main_text, sub_text, var):
		"""
		Builds a generic frame asking for 1 text input
		
		Args:
			master			the parent window
			main_text		main label to display
			sub_text		sub-label to display
			var				Tk variable containing stocking the text input
		"""
		subframe = Tk.LabelFrame(master, text=main_text, padx=10, pady=10,width = 800)
		subframe.pack(fill="both", expand="yes")   
		
		subsubframe = Tk.Frame(subframe, borderwidth=2)
		subsubframe.pack(side=Tk.LEFT, padx=10, pady=10)
		label = Tk.Label(subsubframe, text=sub_text)
		label.pack()
		entry = Tk.Entry(subsubframe, textvariable=var, width=15)
		entry.pack()      

	def build_2subframe(self, master, main_text, sub_text1, sub_text2, var1, var2, align=Tk.RIGHT):
		"""
		Builds a generic frame asking for 2 text inputs
		
		Args:
			master			the parent window
			main_text		main label to display
			sub_text*		sub-labels to display
			var*			Tk variable containing stocking the * text input
			align			Tk alignment of the 2nd input
		"""
		subframe = Tk.LabelFrame(master, text=main_text, padx=10, pady=10,width = 800)
		subframe.pack(fill="both", expand="yes")   
		
		subsubframe_1 = Tk.Frame(subframe, borderwidth=2)
		subsubframe_1.pack(side=Tk.LEFT, padx=10, pady=10)
		label_1 = Tk.Label(subsubframe_1, text=sub_text1)
		label_1.pack()
		entry_1 = Tk.Entry(subsubframe_1, textvariable=var1, width=15)
		entry_1.pack()      

		subsubframe_2 = Tk.Frame(subframe, borderwidth=2)
		subsubframe_2.pack(side=align, padx=10, pady=10)
		label_2 = Tk.Label(subsubframe_2, text=sub_text2)
		label_2.pack()
		entry_2 = Tk.Entry(subsubframe_2, textvariable=var2, width=15)
		entry_2.pack()
		
	def build_3subframe(self, master, main_text, sub_text1, sub_text2, sub_text3, var1, var2, var3):
		"""
		Builds a generic frame asking for 3 text inputs
		
		Args:
			master			the parent window
			main_text		main label to display
			sub_text*		sub-labels to display
			var*			Tk variable containing stocking the * text input
		"""
		subframe = Tk.LabelFrame(master, text=main_text, padx=10, pady=10,width = 800)
		subframe.pack(fill="both", expand="yes")   
		
		subsubframe_1 = Tk.Frame(subframe, borderwidth=2)
		subsubframe_1.pack(side=Tk.LEFT, padx=10, pady=10)
		label_1 = Tk.Label(subsubframe_1, text=sub_text1)
		label_1.pack()
		entry_1 = Tk.Entry(subsubframe_1, textvariable=var1, width=15)
		entry_1.pack()      

		subsubframe_2 = Tk.Frame(subframe, borderwidth=2)
		subsubframe_2.pack(side=Tk.LEFT, padx=10, pady=10)
		label_2 = Tk.Label(subsubframe_2, text=sub_text2)
		label_2.pack()
		entry_2 = Tk.Entry(subsubframe_2, textvariable=var2, width=15)
		entry_2.pack()
		
		subsubframe_3 = Tk.Frame(subframe, borderwidth=2)
		subsubframe_3.pack(side=Tk.LEFT, padx=10, pady=10)
		label_3 = Tk.Label(subsubframe_3, text=sub_text3)
		label_3.pack()
		entry_3 = Tk.Entry(subsubframe_3, textvariable=var3, width=15)
		entry_3.pack()
		
	def get_data(self):
		"""
		"""
		self.temp_hf_splitting1 = self.var_hf_splitting1.get()
		self.temp_hf_splitting2 = self.var_hf_splitting2.get()		
		self.temp_gf1 = self.var_g_factor1.get()
		self.temp_gf2 = self.var_g_factor2.get()		
		self.temp_mu1 = self.var_magn_moment1.get()
		self.temp_mu2 = self.var_magn_moment2.get()		
		self.temp_I1 = self.var_nucl_spin1.get()
		self.temp_I2 = self.var_nucl_spin2.get()		
		self.temp_Es = self.var_lbs_energy1.get()
		self.temp_Et = self.var_lbs_energy2.get()		
		self.temp_ml = self.var_orbital_proj.get()		
		self.temp_mf1 = self.var_hf_state1.get()
		self.temp_mf2 = self.var_hf_state2.get()
		self.temp_B_min = self.var_B_min.get()
		self.temp_B_max = self.var_B_max.get()
		self.temp_B_step = self.var_B_step.get()
		
		self.check_data()
		
		
	def check_data(self):
		"""
		"""
		all_right = True
		error_msg = ''
		try:
			test_hf_splitting1 = float(Fraction(self.temp_hf_splitting1.split()[-1]))
			test_hf_splitting2 = float(Fraction(self.temp_hf_splitting2.split()[-1]))

			test_gf1 = float(Fraction(self.temp_gf1.split()[-1]))
			test_gf2 = float(Fraction(self.temp_gf2.split()[-1]))

			test_mu1 = float(Fraction(self.temp_mu1.split()[-1]))
			test_mu2 = float(Fraction(self.temp_mu2.split()[-1]))
		
			test_I1 = float(Fraction(self.temp_I1.split()[-1]))
			test_I2 = float(Fraction(self.temp_I2.split()[-1]))
			
			test_Es = float(Fraction(self.temp_Es.split()[-1]))
			test_Et = float(Fraction(self.temp_Et.split()[-1]))
			
			test_ml = float(Fraction(self.temp_ml.split()[-1]))
			
			test_mf1 = float(Fraction(self.temp_mf1.split()[-1]))
			test_mf2 = float(Fraction(self.temp_mf2.split()[-1]))
			
			test_B_min  = float(Fraction(self.temp_B_min.split()[-1])) 
			test_B_max  = float(Fraction(self.temp_B_max.split()[-1])) 
			test_B_step = float(Fraction(self.temp_B_step.split()[-1]))
		except:
			self.error_window = Tk.Toplevel(self.root)
			label_warning_1= Tk.Label(self.error_window,justify=Tk.LEFT, text = \
			"It looks like there is something wrong with your input... \n" + \
			"   - Please make sure you only use floats and fractions (the characters allowed are {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, '.', '/'} ) \n" +\
			"   - Check for special characters in your input (newline, tabulation..) \n" +\
			"   - If you don't find any mistake, you can close the main window and try again without an input file (you will be able to build a new one)")
			label_warning_1.pack()
			
			quit_button = Tk.Button(self.error_window, text="Ok", command = self.error_window.destroy)
			quit_button.pack()
			
			self.error_window.mainloop()
			
		if test_I1<0:
			all_right = False
			error_msg += 'ERROR: I_1 is expected to be positive \n\n'

		if test_I2<0:
			all_right = False
			error_msg += 'ERROR: I_2 is expected to be positive \n\n'
			
		if test_ml!=0:
			all_right = False
			error_msg += 'ERROR: This version of the code only works with ML = 0 \n\n'

		if test_Es>0:
			all_right = False
			error_msg += 'ERROR: The energy of the singlet state is expected to be negative \n\n'

		if test_Et>0:
			all_right = False
			error_msg += 'ERROR: The energy of the triplet state is expected to be negative \n\n'

		if test_B_min<0 or test_B_max<0 or test_B_step<0:
			all_right = False
			error_msg += 'ERROR: The magnetic field and the step are expected to be positive \n\n'
			
		if test_B_step==0:
			all_right = False
			error_msg += 'ERROR: The step of the magnetic field cannot be 0 \n\n'

		if test_B_min > test_B_max:
			all_right = False
			error_msg += 'ERROR: Wrong range for the magnetic field. You entered B_min > B_max \n\n'	
			
		if np.abs(test_mf1)>test_I1+.5:
			all_right = False
			error_msg += 'ERROR: abs(MF_1) > I1 + 1/2 is not possible \n\n'
			
		if np.abs(test_mf2)>test_I2+.5:
			all_right = False
			error_msg += 'ERROR: abs(MF_2) > I2 + 1/2 is not possible \n\n'
			
		if not np.mod(test_mf1,1) == np.mod(test_I1+.5,1):
			all_right = False
			error_msg += 'ERROR: MF_1 and I_1 + 1/2 must have the same fractional part \n\n'
			
		if not np.mod(test_mf2,1) == np.mod(test_I2+.5,1):
			all_right = False
			error_msg += 'ERROR: MF_2 and I_2 + 1/2 must have the same fractional part \n\n'

		if all_right :
			self.assign()
			self.save_input()
		else :
			self.warning_window = Tk.Toplevel(self.root)
			label_warning_1= Tk.Label(self.warning_window,justify=Tk.LEFT, text = "Please correct the following errors: \n\n")
			label_warning_1.pack()
			label_warning_2= Tk.Label(self.warning_window,justify=Tk.LEFT,fg = 'red', text = error_msg)
			label_warning_2.pack()
			
			quit_button = Tk.Button(self.warning_window, text="Ok", command = self.warning_window.destroy)
			quit_button.pack()
			
			self.warning_window.mainloop()

	def assign(self):
		"""
		"""
		self.simulation.hf_splitting1 = float(Fraction(self.temp_hf_splitting1.split()[-1]))
		self.simulation.hf_splitting2 = float(Fraction(self.temp_hf_splitting2.split()[-1]))

		self.simulation.gf1 = float(Fraction(self.temp_gf1.split()[-1]))
		self.simulation.gf2 = float(Fraction(self.temp_gf2.split()[-1]))

		self.simulation.gamma1 = self.simulation.hf_splitting1 * self.simulation.gf1 / MHZ_TO_WAVENUMBERS / WAVENUMBERS_TO_HARTREE
		self.simulation.gamma2 = self.simulation.hf_splitting2 * self.simulation.gf2 / MHZ_TO_WAVENUMBERS / WAVENUMBERS_TO_HARTREE

		self.simulation.mu1 = float(Fraction(self.temp_mu1.split()[-1]))
		self.simulation.mu2 = float(Fraction(self.temp_mu2.split()[-1]))

		self.simulation.I1 = float(Fraction(self.temp_I1.split()[-1]))
		self.simulation.I2 = float(Fraction(self.temp_I2.split()[-1]))

		self.simulation.Es = float(Fraction(self.temp_Es.split()[-1]))  / WAVENUMBERS_TO_HARTREE
		self.simulation.Et = float(Fraction(self.temp_Et.split()[-1]))  / WAVENUMBERS_TO_HARTREE

		self.simulation.ml = float(Fraction(self.temp_ml.split()[-1]))

		self.simulation.mf1 = float(Fraction(self.temp_mf1.split()[-1]))
		self.simulation.mf2 = float(Fraction(self.temp_mf2.split()[-1]))

		self.simulation.B_min  = float(Fraction(self.temp_B_min.split()[-1]))  / GAUSS_TO_TESLA
		self.simulation.B_max  = float(Fraction(self.temp_B_max.split()[-1]))  / GAUSS_TO_TESLA
		self.simulation.B_step = float(Fraction(self.temp_B_step.split()[-1]))  / GAUSS_TO_TESLA

		self.simulation.B_grid = np.arange(self.simulation.B_min, self.simulation.B_max + self.simulation.B_step, self.simulation.B_step)
		
	def save_input(self):
		"""
		"""
		self.save_window = Tk.Toplevel(self.root)
		
		label_save= Tk.Label(self.save_window,justify=Tk.LEFT, text= \
		" If you want to save these parameters as an input file, write the name of the file bellow (e.g. 'my_input.txt').")
		label_save.pack()
		
		self.var_save_file = Tk.StringVar()
		self.var_save_file.set('my_input.txt')
		entry_save_file = Tk.Entry(self.save_window, textvariable=self.var_save_file, width=100)
		entry_save_file.pack()
		
		subframe=Tk.Frame(self.save_window)
		subframe.pack()

		save_button = Tk.Button(subframe, text="Save", command = self.check_save_file)
		save_button.pack(side=Tk.RIGHT)
		
		quit_button = Tk.Button(subframe, text="Don't Save", command = self.exit_interface)
		quit_button.pack(side=Tk.RIGHT)

		self.save_window.mainloop()

	def check_save_file(self):
		"""
		"""
		self.file_name = self.var_save_file.get()
		self.file_path = os.path.join(self.simulation.path,os.path.normcase(self.file_name))
		
		if os.path.isfile(self.file_path):
			self.confirm_window = Tk.Toplevel(self.root)
			message = Tk.Label(self.confirm_window, justify=Tk.LEFT, text= \
			" The file \n '" + self.file_path + "' \n already exists. \n Do you want to overwrite it?")
			message.pack()
			back_button = Tk.Button(self.confirm_window, text="Go Back", command = self.confirm_window.destroy)
			back_button.pack(side="left")
			build_button = Tk.Button(self.confirm_window, text="Overwrite", command = self.save_to_file)
			build_button.pack(side="right")	
		else:
			self.save_to_file()
		
	def save_to_file(self):
		"""
		"""
		text_file = open(self.file_path, "w")
		text_file.write("ATOMIC CONSTANTS AND ENERGIES\n=============================\n\n\tHyperfine Splittings (MHz):\n\t\tAtom_1\t\t"               \
						+ self.temp_hf_splitting1                                                                                                     \
						+ "\n\t\tAtom_2\t\t"                                                                                                          \
						+ self.temp_hf_splitting2                                                                                                     \
						+ "\n\n\tG-Factors:\n\t\tgF_1\t\t"                                                                                            \
						+ self.temp_gf1                                                                                                               \
						+ "\n\t\tgF_2\t\t"                                                                                                            \
						+ self.temp_gf2                                                                                                               \
						+ "\n\n\tNuclear Magnetic Moments (Nuclear Bohr Magnetons):\n\t\tmu_1\t\t"                                                    \
						+ self.temp_mu1                                                                                                               \
						+ "\n\t\tmu_2\t\t"                                                                                                            \
						+ self.temp_mu2                                                                                                               \
						+ "\n\n\tNuclear Spins:\n\t\tI_1\t\t\t"                                                                                       \
						+ self.temp_I1                                                                                                                \
						+ "\n\t\tI_2\t\t\t"                                                                                                           \
						+ self.temp_I2                                                                                                                \
						+ "\n\n\tLeast Bound State Energy (cm^-1):\n\t\tSinglet\t\t"                                                                  \
						+ self.temp_Es                                                                                                                \
						+ "\n\t\tTriplet \t"                                                                                                          \
						+ self.temp_Et                                                                                                                \
						+ "\n\n\nINPUT STATE AND MAGNETIC FIELD RANGE\n====================================\n\n\tOrbital Projection:\n\t\tmL\t\t\t"   \
						+ self.temp_ml                                                                                                                \
						+ "\n\n\tInput State:\n\t\tMF_1\t\t"                                                                                          \
						+ self.temp_mf1                                                                                                               \
						+ "\n\t\tMF_2\t\t"                                                                                                            \
						+ self.temp_mf2                                                                                                               \
						+ "\n\n\tMagnetic Field Range and Step Size (Gauss):\n\t\tMinimum Magnetic Field\t\t\t"                                       \
						+ self.temp_B_min                                                                                                             \
						+ "\n\t\tMaximum Magnetic Field\t\t\t"                                                                                        \
						+ self.temp_B_max                                                                                                             \
						+ "\n\t\tMagnetic Field Step Size\t\t"                                                                                        \
						+ self.temp_B_step		                                                                                                      \
						)
		text_file.close()
		self.exit_interface()

	def exit_interface(self):
		"""
		"""
		self.save_window.destroy()
		self.root.destroy()
	
	def user_close(self):
		"""
		"""
		self.root.destroy()
		sys.exit("You closed the main application")
		
class InterfacePlot(object):
	"""
	GUI interface built with the Tkinter module (Tk).
	Asks the user for a work directory and an optional input file.
	Can construct a work directory if necessary.
	
	Note:
		When main() is executed, this class modifies the attributes of the global variable 'my_simulation'.
	"""
	def __init__(self,root,simulation_object):
		"""
		Main Tk application.
		Asks the user for a work directory and an optional input file.
		
		Calls:
			InterfacePath.check_input 		when pressing 'Done' 
			InterfacePath.user_close		if the user closes the main window (--> sys.exit)			
		"""
		self.root=root
		self.simulation=simulation_object
		
		self.status=[Tk.DISABLED, Tk.NORMAL]
		
		self.label_top = Tk.Label(self.root,justify=Tk.LEFT, text= \
		" Select the results you want to plot")
		self.label_top.pack()
		
		self.main_frame=Tk.Frame(self.root)
		self.main_frame.pack()
		
		self.frame_fig_1 = Tk.Frame(self.main_frame, borderwidth=2, relief=Tk.GROOVE)
		self.frame_fig_1.pack(side=Tk.LEFT)
		
		
		self.var_fig_1 = Tk.IntVar()
		self.var_fig_1.set(1)
		self.button_fig_1 = Tk.Checkbutton(self.frame_fig_1, text = \
		'FIG 1: STATES ENERGIES AND PREDICTED RESONANCES', variable = self.var_fig_1, command = self.update_buttons_fig_1)
		self.button_fig_1.pack()
		

		self.subframe_threshold_fig_1 = Tk.LabelFrame(self.frame_fig_1, text='Threshold States', padx=10, pady=10)
		self.subframe_threshold_fig_1.pack(fill="both", expand="yes")
		
		self.vars_threshold_fig_1 = [Tk.IntVar() for i in range(my_simulation.num_ass)]
		self.vars_threshold_fig_1[0].set(1)
		self.buttons_threshold_fig_1=[]
		for i in range (my_simulation.num_ass):
			self.buttons_threshold_fig_1 += [Tk.Checkbutton(self.subframe_threshold_fig_1, text = \
				'Threshold state {} : F1 = '.format(i+1) + str(Fraction(my_simulation.hf_states_ass[i,0])) + \
				',  F2 = ' + str(Fraction(my_simulation.hf_states_ass[i,1])),variable = self.vars_threshold_fig_1[i])]
			self.buttons_threshold_fig_1[i].pack()
		
		self.var_bound_fig_1 = Tk.IntVar()
		self.var_bound_fig_1.set(1)
		self.button_bound_fig_1 = Tk.Checkbutton(self.frame_fig_1, text = 'Bound states', \
						variable = self.var_bound_fig_1)
		self.button_bound_fig_1.pack()
		
		self.var_crossings_fig_1 = Tk.IntVar()
		self.var_crossings_fig_1.set(1)
		self.button_crossings_fig_1 = Tk.Checkbutton(self.frame_fig_1, text = 'Predicted resonnances', \
						variable = self.var_crossings_fig_1)
		self.button_crossings_fig_1.pack()
		
		
		self.frame_fig_2 = Tk.Frame(self.main_frame, borderwidth=2, relief=Tk.GROOVE)
		self.frame_fig_2.pack(side=Tk.RIGHT)
		
		
		self.var_fig_2 = Tk.IntVar()
		self.var_fig_2.set(1)
		self.button_fig_2 = Tk.Checkbutton(self.frame_fig_2, text = \
		'FIG 2: STATES PROJECTIONS ON SINGLET SUBSPACE', variable = self.var_fig_2, command = self.update_buttons_fig_2)
		self.button_fig_2.pack()

		self.subframe_threshold_fig_2 = Tk.LabelFrame(self.frame_fig_2, text='Threshold States', padx=10, pady=10)
		self.subframe_threshold_fig_2.pack(fill="both", expand="yes")		
		
		self.vars_threshold_fig_2 = [Tk.IntVar() for i in range(my_simulation.num_ass)]
		self.vars_threshold_fig_2[0].set(1)
		self.buttons_threshold_fig_2=[]
		for i in range (my_simulation.num_ass):
			self.buttons_threshold_fig_2 += [Tk.Checkbutton(self.subframe_threshold_fig_2, text = \
				'Threshold state {} : F1 = '.format(i+1) + str(Fraction(my_simulation.hf_states_ass[i,0])) + \
				',  F2 = ' + str(Fraction(my_simulation.hf_states_ass[i,1])),variable = self.vars_threshold_fig_2[i])]
			self.buttons_threshold_fig_2[i].pack()
		
		
		self.var_bound_fig_2 = Tk.IntVar()
		self.var_bound_fig_2.set(1)
		self.button_bound_fig_2 = Tk.Checkbutton(self.frame_fig_2, text = 'Bound states', \
						variable = self.var_bound_fig_2)
		self.button_bound_fig_2.pack()
		

		self.var_keep_same_fig = Tk.IntVar()
		self.button_keep_same_fig = Tk.Checkbutton(self.root, text = \
		'Do NOT create a new figure for the plot \n'+ \
		'(Plots the result in the current matplotlib figure)', variable = self.var_keep_same_fig, state = Tk.DISABLED)
		self.button_keep_same_fig.pack()
		
		self.label_bottom = Tk.Label(self.root,justify=Tk.LEFT, text= \
		" If you select nothing, you can still access the attributes of your simulation in the terminal. \n"+\
		" To do so, enter my_simulation.{the attribute you want to access}, without the braces {}")
		self.label_bottom.pack()
		
		self.quit_button = Tk.Button(self.root, text="Done", command = self.plot_simulation)
		self.quit_button.pack()
		
		
		self.root.protocol("WM_DELETE_WINDOW", self.user_close)
		self.root.mainloop()
		
	def update_buttons_fig_1(self):
		self.var_bound_fig_1.set(0)
		self.button_bound_fig_1.config(state=self.status[self.var_fig_1.get()])
	

		for i in range (my_simulation.num_ass):
			self.vars_threshold_fig_1[i].set(0)
			self.buttons_threshold_fig_1[i].config(state=self.status[self.var_fig_1.get()])
		
		self.var_crossings_fig_1.set(0)
		self.button_crossings_fig_1.config(state=self.status[self.var_fig_1.get()])
		
		self.var_keep_same_fig.set(0)
		self.button_keep_same_fig.config(state=self.status[not (self.var_fig_1.get() and self.var_fig_2.get())])
	
	def update_buttons_fig_2(self):
		self.var_bound_fig_2.set(0)
		self.button_bound_fig_2.config(state=self.status[self.var_fig_2.get()])	
		
		for i in range (my_simulation.num_ass):
			self.vars_threshold_fig_2[i].set(0)
			self.buttons_threshold_fig_2[i].config(state=self.status[self.var_fig_2.get()])
			
		self.var_keep_same_fig.set(0)
		self.button_keep_same_fig.config(state=self.status[not (self.var_fig_1.get() and self.var_fig_2.get())])
	
	def plot_simulation(self):
		"""
		"""
		self.root.destroy()
		my_simulation.plot_result( \
			bool(self.var_fig_1.get()), \
			bool(self.var_fig_2.get()), \
			bool(self.var_keep_same_fig.get()), \
			bool(self.var_bound_fig_1.get()), \
			[bool(self.vars_threshold_fig_1[i].get()) for i in range (my_simulation.num_ass)], \
			bool(self.var_crossings_fig_1.get()), \
			bool(self.var_bound_fig_2.get()), \
			[bool(self.vars_threshold_fig_2[i].get()) for i in range (my_simulation.num_ass)], \
			)
	
	def user_close(self):
		"""
		"""
		self.root.destroy()
		sys.exit("You closed the main application")

if __name__ == '__main__': main()	
	