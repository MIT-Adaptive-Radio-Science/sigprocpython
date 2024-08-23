"""
Create a chirp  and test the PFB.

"""
from pathlib import Path
import numpy as np
import scipy.signal as sig
from mitarspysigproc import pfb_decompose, pfb_reconstruct,kaiser_coeffs,kaiser_syn_coeffs,npr_analysis,npr_synthesis,rref_coef
import matplotlib.pyplot as plt


def create_chirp(t_len, fs, bw,pad):
    """
    """
    t = np.linspace(0,t_len, int(t_len*fs))

    x = sig.chirp(t,t1=t_len,f0=0,f1=bw,method='linear')

    xout = np.concatenate((pad[0],x,pad[1]),axis=0)
    tp1 = -1*np.arange(0,len(pad[0]),dtype=float)[::-1]/fs
    tp2 = np.arange(0,len(pad[1]),dtype=float)/fs + t_len
    tout  = np.concatenate((tp1,t,tp2),axis=0)

    return tout,xout


def runchirptest(t_len,fs,bw,nzeros,nchans):
    """

    Parameters
    ----------

    Returns
    -------

    """
    pad = [np.zeros(nzeros),np.zeros(nzeros)]
    t,x = create_chirp(t_len,fs,bw,pad)
    mainpath = Path(__file__).resolve().parent.parent
    # fname = mainpath.joinpath('coeffs',"kaiseranalysis{:04d}.csv".format(nchans))
    coeffs = kaiser_coeffs(nchans)
    mask = np.ones(nchans,dtype=bool)
    xout = pfb_decompose(x, nchans, coeffs, mask)
    fillmethod = ''
    fillparams = [0,0]
    # fname = mainpath.joinpath('coeffs',"kaisersynthesis{:04d}.csv".format(nchans))
    syn_coeffs = kaiser_syn_coeffs(nchans)
    x_rec = pfb_reconstruct(xout,nchans,syn_coeffs,mask,fillmethod,fillparams=[],realout=True)
    return x_rec,t,x, xout

def runnprchirptest(t_len,fs,bw,nzeros,nchans):
    """

    Parameters
    ----------

    Returns
    -------

    """
    pad = [np.zeros(nzeros),np.zeros(nzeros)]
    t,x = create_chirp(t_len,fs,bw,pad)
    mainpath = Path(__file__).resolve().parent.parent
    # fname = mainpath.joinpath('coeffs',"kaiseranalysis{:04d}.csv".format(nchans))
    coeffs = rref_coef(nchans,64)
    mask = np.ones(nchans,dtype=bool)
    xout = npr_analysis(x, nchans, coeffs)
    fillmethod = ''
    fillparams = [0,0]
    # fname = mainpath.joinpath('coeffs',"kaisersynthesis{:04d}.csv".format(nchans))
    x_rec = npr_synthesis(xout,nchans,coeffs)
    return x_rec,t,x, xout


def nexpow2(x):
    """Returns the next power of two.

    Parameters
    ----------
    x : int
        Inital number.

    Returns
    -------
    int
        The next power of two of x.
    """

    return int(np.power(2,np.ceil(np.log2(x))))

def plotdata(inchirp,outchirp,tin,tout):
    """Plot the data and return the figure.


    """

    fig, ax = plt.subplots(2, 1, figsize=(10,5))

    inlen = inchirp.shape[0]
    outlen = outchirp.shape[0]
    tau =  tin[1]-tin[0]

    ax[0].plot(tin,inchirp,label='Input')
    ax[0].plot(tout,outchirp,label='Output')

    ax[0].set_xlabel('Time in Seconds')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_title('Time Domain')
    ax[0].grid(True)

    nfft_in = nexpow2(inlen)
    nfft_out = nexpow2(outlen)

    in_freq = np.fft.fftshift(np.fft.fftfreq(nfft_in,d=tau))
    out_freq = np.fft.fftshift(np.fft.fftfreq(nfft_out,d=tau))

    spec_in = np.abs(np.fft.fftshift(np.fft.fft(inchirp,n=nfft_in)))**2
    spec_out = np.abs(np.fft.fftshift(np.fft.fft(outchirp[:,0],n=nfft_out)))**2

    spec_in_log = 10*np.log10(spec_in)
    spec_out_log = 10*np.log10(spec_out)

    ax[1].plot(in_freq,spec_in_log,label='Input')
    ax[1].plot(out_freq,spec_out_log,label='Output')

    ax[1].set_xlabel('Frequency in Hz')
    ax[1].set_ylabel('Amp dB')
    ax[1].set_title('Frequency Content')
    ax[1].grid(True)
    ax[1].set_ylim([0,60])
    fig.tight_layout()
    return fig

def runexample():
    """ """
    nchans = 32
    fs = 10000
    t_len = 6.5536
    
    bw = 2000
    nzeros = 1024
    

    x_rec,t,x,_ = runchirptest(t_len,fs,bw,nzeros,nchans)
    x_rec = x_rec[:len(x)]

    fig = plotdata(x,x_rec,t,t)
    fig.savefig('chirpdata.png')
    plt.close(fig)

    x_rec,t,x,_ = runnprchirptest(t_len,fs,bw,nzeros,nchans)
    x_rec = x_rec[:len(x),np.newaxis] # need to add new axis due to plotting issue

    fig = plotdata(x,x_rec,t,t)
    fig.savefig('chirpdatanpr.png')
    plt.close(fig)
if __name__ == "__main__":
    runexample()
