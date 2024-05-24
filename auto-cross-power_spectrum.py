import numpy as np
import healpy as hp
import pymaster as nmt

nside = 512 # Healpix resolution of your maps 


def compute_master(f_a, f_b, wsp):
    cl_coupled = nmt.compute_coupled_cell(f_a, f_b)
    cl_decoupled = wsp.decouple_cell(cl_coupled)
    return cl_decoupled

# Function that give us the auto-spectrum for a map
def auto_spectrum(mask, map, lmax, dl, purify_e=False, purify_b=False, beam=None):

	#Create field spin0
	f0 = nmt.NmtField(mask, [map[0,:]], lmax_sht=lmax, beam=beam)
	#Create field spin2
	f2 = nmt.NmtField(mask, [map[1,:],map[2,:]], lmax_sht=lmax, purify_e=purify_e, purify_b=purify_b, beam=beam)
                
	#Create binning scheme
	b = nmt.NmtBin(nside, nlb=dl, lmax=lmax)

	#Generate Workspace object, that is based only on the masks of the objects to 
    #correlate (This computes the coupling matrix that will be applied to all 
    #the power spectra)
	w00 = nmt.NmtWorkspace()
	w00.compute_coupling_matrix(f0, f0, b)

	w02 = nmt.NmtWorkspace()
	w02.compute_coupling_matrix(f0, f2, b)

	w22 = nmt.NmtWorkspace()
	w22.compute_coupling_matrix(f2, f2, b)

	# Compute the power spectrum of our two input fields
	cl_master_tt = compute_master(f0, f0, w00) # TT
	cl_master_tetb = compute_master(f0, f2, w02) # TE TB
	cl_master_eb = compute_master(f2, f2, w22) # EE EB BE BB
            
	#plot power spectra
	cl_tt =  cl_master_tt[0] #label='TT '
	cl_te = cl_master_tetb[0] #label='TE '
	cl_tb = cl_master_tetb[1] #label='TB '
	cl_ee = cl_master_eb[0] #label='EE '
	cl_eb = cl_master_eb[1] #label='EB '
	cl_bb = cl_master_eb[3] #label='BB ' 

	power_spectrum_modes = np.array([b.get_effective_ells(), cl_tt, cl_ee, cl_bb, cl_te, cl_tb, cl_eb])
    
	return power_spectrum_modes # ell, TT, EE, BB, TE, TB, EB

# Function that give us the cross-spectrum between two maps
def cross_spectrum(mask, map_1, map_2, lmax, dl, purify_e=False, purify_b=False, beam=None):

	#Create fields spin-0 and spin-2 for both maps
	f0_1 = nmt.NmtField(mask, [map_1[0,:]], lmax_sht=lmax, beam=beam)
	f2_1 = nmt.NmtField(mask, [map_1[1,:],map_1[2,:]], lmax_sht=lmax, purify_e=purify_e, purify_b=purify_b, beam=beam)
    
	f0_2 = nmt.NmtField(mask, [map_2[0,:]], lmax_sht=lmax, beam=beam)
	f2_2 = nmt.NmtField(mask, [map_2[1,:],map_2[2,:]], lmax_sht=lmax, purify_e=purify_e, purify_b=purify_b, beam=beam)
                      
	#Create binning scheme
	b = nmt.NmtBin(nside, nlb=dl, lmax=lmax)

	#Generate Workspace object, that is based only on the masks of the objects to 
    #correlate (This computes the coupling matrix that will be applied to all 
    #the power spectra)

	w00 = nmt.NmtWorkspace()
	w02 = nmt.NmtWorkspace()
	w22 = nmt.NmtWorkspace()
      
	w00.compute_coupling_matrix(f0_1, f0_1, b)
	w02.compute_coupling_matrix(f0_1, f2_1, b)
	w22.compute_coupling_matrix(f2_1, f2_1, b)
      
    # Compute cross-spectrum
	cl_master_t1_t2 = compute_master(f0_1, f0_2, w00) # TT
	cl_master_eb_12 = compute_master(f2_1, f2_2, w22) # EE EB BE BB
    
	# Plot cross-spectra
	cl_t1_t2 = cl_master_t1_t2[0]
	cl_e1_e2 = cl_master_eb_12[0]
	cl_e1_b2 = cl_master_eb_12[1]
	cl_b1_e2 = cl_master_eb_12[2]
	cl_b1_b2 = cl_master_eb_12[3]

    # We add a zeros array in order to mantain the same shape as the same shape as for the auto-spectra
	power_spectrum_modes = np.array([b.get_effective_ells(), cl_t1_t2, cl_e1_e2, cl_b1_b2, cl_e1_b2, cl_b1_e2, np.zeros(int(2*nside/dl))])
    
	return power_spectrum_modes # ell, T1T2, E1E2, B1B2, E1B2, E2B1, null 