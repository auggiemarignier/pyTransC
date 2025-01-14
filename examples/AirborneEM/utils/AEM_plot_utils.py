import numpy
import matplotlib.pyplot

def read_chain(fp):
    fh=open(fp)
    lines=fh.readlines()
    chain=[]
    for line in lines:
        fields=line.split()
        msft=float(fields[0])
        n=int(fields[1])
        d=numpy.array(fields[2:n+2],dtype=float)
        c=numpy.array(fields[n+2:n+2+n],dtype=float)
        chain.append([n,msft,d,c])
    return chain

def plot_sample(smpl,zmax):
    n,msft,d,c=smpl
    depth=[d[0]]
    resistivity=[1/10**c[0]]
    for i in range(1,n):
        depth.append(d[i])
        depth.append(d[i])
        resistivity.append(1/10**c[i-1])
        resistivity.append(1/10**c[i])
    depth.append(zmax*2)
    resistivity.append(1/10**c[-1])

    matplotlib.pyplot.figure(figsize=(2,4))
    matplotlib.pyplot.gca().invert_yaxis()
    matplotlib.pyplot.plot(resistivity,depth)
    matplotlib.pyplot.ylabel('depth (m)')
    matplotlib.pyplot.xlabel('resistvity (Ω m)')
    matplotlib.pyplot.ylim([zmax,0])
    return 

def plot_mean(fn,zmax):
    fh=open(fn, "r")
    lines=fh.readlines()
    fields=lines[0].split()
    x=float(fields[6])-361000
    y=float(fields[7])
    elev=float(fields[8])
    depth=numpy.array(fields[17:137],dtype=float)
    
    mean=1.0/numpy.array(fields[137:257],dtype=float)
    mode=1.0/numpy.array(fields[257:377],dtype=float)
    p50=1.0/numpy.array(fields[377:497],dtype=float)
    p10=1.0/numpy.array(fields[497:617],dtype=float)
    p90=1.0/numpy.array(fields[617:737],dtype=float)
    maxlk=1.0/numpy.array(fields[737:857],dtype=float)
    bestfit=1.0/numpy.array(fields[857:977],dtype=float)
    cp=numpy.array(fields[977:1097],dtype=float)

    depth=-depth

    matplotlib.pyplot.figure(figsize=(2,4))
    matplotlib.pyplot.gca().invert_yaxis()
    matplotlib.pyplot.plot(mean,-depth)
    matplotlib.pyplot.ylabel('depth (m)')
    matplotlib.pyplot.xlabel('resistvity (Ω m)')
    matplotlib.pyplot.ylim([zmax,0])
    #print('horizontal distance: ',x)

def get_sample(smpl,zmax):
    n,msft,d,c=smpl
    depth=[d[0]]
    resistivity=[1/10**c[0]]
    for i in range(1,n):
        depth.append(d[i])
        depth.append(d[i])
        resistivity.append(1/10**c[i-1])
        resistivity.append(1/10**c[i])
    depth.append(zmax*2)
    resistivity.append(1/10**c[-1])
    return resistivity,depth

def plot_histogram(chain, xmin, xmax, nx, zmin, zmax, nz,cmap='viridis',log=False,
                     cmapmin=0.,cmapmax=1.0,returndata=False,figsize=(2,4),
                     density_cutoff=None,pcentiles=None,interfaceplotcolor=None,
                     density=None,verbose=False,filename=None):

    #xedges = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), nx)
    xedges = numpy.linspace(xmin, xmax, nx)
    yedges = numpy.linspace(zmin, zmax, nz)

    if(density is not None):
        HH = density
    else:
        if(verbose): print('Start density calculation')
        for idx,piece in enumerate(chain):
            n,msft,d,c=piece
            dd=[]
            val=1./10**c[0]
            dd.append([val,d[0]])
            for i in range(1,n):
                val=1./10**c[i-1]
                dd.append([val,d[i]])
                val=1./10**c[i]
                dd.append([val,d[i]])
            dd.append([val,zmax])
            dd=numpy.array(dd)
            xxx = []
            yyy = []
            
            for i in range(len(yedges)-1):
                y = (yedges[i] + yedges[i + 1]) / 2.0
                x = numpy.interp(y, dd[:, 1], dd[:, 0])
                xxx.append(x)
                yyy.append(y)

            for i in range(1, len(dd[:, 0]) - 1, 2):
                x0 = min(dd[i, 0], dd[i + 1, 0])
                x1 = max(dd[i, 0], dd[i + 1, 0])
                for j in range(len(xedges) - 1):
                    x = (xedges[j] + xedges[j + 1]) / 2.0
                    if x >= x0 and x <= x1:
                        xxx.append(x)
                        yyy.append(dd[i, 1])

            H, xedges, yedges = (
                        numpy.histogram2d(xxx, yyy, bins=(xedges, yedges)))

            if (idx==0):
                HH = numpy.clip(H, 0, 1)
            else:
                HH = HH + numpy.clip(H, 0, 1)
                
        if(verbose): print('End density calculation')
        
    hmax = HH.max()
    if(log):
        HHplot = numpy.zeros_like(HH)
        mask = HH != 0
        HHplot[mask] = numpy.log(HH[mask]/hmax) # map colormap to a log of density
        HHplot[~mask] = -numpy.inf
        HHplot = numpy.transpose(HHplot)
    else:
        HHplot = numpy.copy(numpy.transpose(HH))/hmax
    
    if(density_cutoff is not None): # white out all cells with density below n% of maximum 
                                    # (if(log) then density_cutoff = numpy.log(n/100); othrwise density_cutoff=n/100)
        cutoff = density_cutoff/100
        if(log): cutoff = numpy.log(cutoff)
        HHplot[numpy.where(HHplot<=cutoff)] = -numpy.inf
        
    nz,nx = numpy.shape(HHplot)
    X,Y = numpy.meshgrid(numpy.linspace(xmin,xmax,nx),numpy.linspace(0,zmax,nz))

    if(cmapmin != 0.0 or cmapmax != 1.0):
        cmapnew = truncate_colormap(matplotlib.pyplot.get_cmap(cmap), minval=cmapmin, maxval=cmapmax, n=100)
    else:
        cmapnew = cmap

    if(len(chain[0][2])==1): interfaceplotcolor = None
    if(interfaceplotcolor is not None):
        #fig, (ax,ax2) = matplotlib.pyplot.subplots(1,2,figsize=figsize,width_ratios=[3.,1.])
        gs_kw = dict(width_ratios=[3,1])
        fig, (ax,ax2) = matplotlib.pyplot.subplots(ncols=2, nrows=1, figsize=figsize, constrained_layout=True, sharey=True,gridspec_kw=gs_kw)
    else:
        fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
        
    ax.invert_yaxis()
    ax.set_ylabel('depth (m)')
    ax.set_xlabel('resistvity (Ω m)')
    n_levels = 99  # number of contour levels to plot
    ax.contourf(X, Y, HHplot,n_levels,cmap=cmap)
    ax.set_ylim([zmax,0])
    ax.set_xlim([xmin,xmax])
    
    #plot percentile models on top
    z = numpy.linspace(0,zmax,nz)
    if(pcentiles is not None):
        v = numpy.percentile(HH,pcentiles[0],axis=0,method='median_unbiased')
        
        s = numpy.zeros((len(chain),nz))
        for i in range(len(chain)):
            r,d = get_sample(chain[i],500)
            for j,x in enumerate(z):
                s[i,j] = r[d.index(next(obj for obj in d if obj >= x))]
        p = numpy.percentile(s,pcentiles[0],axis=0)
        for i,x in enumerate(p):
            ax.plot(x,z,pcentiles[1][i],lw=pcentiles[2][i])

    # plot density of interfaces
    if(interfaceplotcolor is not None):
        X,Y = numpy.meshgrid(numpy.linspace(xmin,xmax,1),numpy.linspace(0,zmax,nz))
        dz=zmax/nz
        den = numpy.zeros(nz)
        for i in range(len(chain)):
            r,d = get_sample(chain[i],500)
            for j in range(1,len(d)):
                k = int(d[j]/dz)
                if(k< nz): den[k]+=1
        den /= numpy.max(den)
        ax2.fill_betweenx(z, den,color=interfaceplotcolor)

    if(filename is not None): matplotlib.pyplot.savefig(filename)
    
    return HH

def plot_histogram_is(transd_ensemble, xmin, xmax, nx, zmin, zmax, nz,cmap='viridis',log=False,
                     cmapmin=0.,cmapmax=1.0,returndata=False,figsize=(2,4),
                     density_cutoff=None,pcentiles=None,interfaceplotcolor=None,
                     density=None,verbose=False,filename=None):

    #xedges = numpy.logspace(numpy.log10(xmin), numpy.log10(xmax), nx)
    xedges = numpy.linspace(xmin, xmax, nx)
    yedges = numpy.linspace(zmin, zmax, nz)

    if(density is not None):
        HH = density
        idx = 0
        for state,x in enumerate(transd_ensemble):
            idx+= len(transd_ensemble[state])
    else:
        if(verbose): print('Start density calculation')
        #for idx,piece in enumerate(chain):
        idx = 0
        for state,x in enumerate(transd_ensemble):
            n = state
            for i,y in enumerate(x):
                d = [0., *y[:n]]
                c = y[n:]
                
                dd=[]
                val=1./10**c[0]
                dd.append([val,d[0]])
                for i in range(1,n):
                    val=1./10**c[i-1]
                    dd.append([val,d[i]])
                    val=1./10**c[i]
                    dd.append([val,d[i]])
                dd.append([val,zmax])
                dd=numpy.array(dd)
                xxx = []
                yyy = []
            
                for i in range(len(yedges)-1):
                    y = (yedges[i] + yedges[i + 1]) / 2.0
                    x = numpy.interp(y, dd[:, 1], dd[:, 0])
                    xxx.append(x)
                    yyy.append(y)

                for i in range(1, len(dd[:, 0]) - 1, 2):
                    x0 = min(dd[i, 0], dd[i + 1, 0])
                    x1 = max(dd[i, 0], dd[i + 1, 0])
                    for j in range(len(xedges) - 1):
                        x = (xedges[j] + xedges[j + 1]) / 2.0
                        if x >= x0 and x <= x1:
                            xxx.append(x)
                            yyy.append(dd[i, 1])

                H, xedges, yedges = (
                            numpy.histogram2d(xxx, yyy, bins=(xedges, yedges)))

                if (idx==0):
                    HH = numpy.clip(H, 0, 1)
                else:
                    HH = HH + numpy.clip(H, 0, 1)
                idx+=1
                
        if(verbose): print('End density calculation')
        
    hmax = HH.max()
    if(log):
        HHplot = numpy.zeros_like(HH)
        mask = HH != 0
        HHplot[mask] = numpy.log(HH[mask]/hmax) # map colormap to a log of density
        HHplot[~mask] = -numpy.inf
        HHplot = numpy.transpose(HHplot)
    else:
        HHplot = numpy.copy(numpy.transpose(HH))/hmax
    
    if(density_cutoff is not None): # white out all cells with density below n% of maximum 
                                    # (if(log) then density_cutoff = numpy.log(n/100); othrwise density_cutoff=n/100)
        cutoff = density_cutoff/100
        if(log): cutoff = numpy.log(cutoff)
        HHplot[numpy.where(HHplot<=cutoff)] = -numpy.inf
        
    nz,nx = numpy.shape(HHplot)
    X,Y = numpy.meshgrid(numpy.linspace(xmin,xmax,nx),numpy.linspace(0,zmax,nz))

    if(cmapmin != 0.0 or cmapmax != 1.0):
        cmapnew = truncate_colormap(matplotlib.pyplot.get_cmap(cmap), minval=cmapmin, maxval=cmapmax, n=100)
    else:
        cmapnew = cmap

    #if(len(chain[0][2])==1): interfaceplotcolor = None
    if(interfaceplotcolor is not None):
        #fig, (ax,ax2) = matplotlib.pyplot.subplots(1,2,figsize=figsize,width_ratios=[3.,1.])
        gs_kw = dict(width_ratios=[3,1])
        fig, (ax,ax2) = matplotlib.pyplot.subplots(ncols=2, nrows=1, figsize=figsize, constrained_layout=True, sharey=True,gridspec_kw=gs_kw)
    else:
        fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
        
    ax.invert_yaxis()
    ax.set_ylabel('depth (m)')
    ax.set_xlabel('resistvity (Ω m)')
    n_levels = 99  # number of contour levels to plot
    ax.contourf(X, Y, HHplot,n_levels,cmap=cmap)
    ax.set_ylim([zmax,0])
    ax.set_xlim([xmin,xmax])
    
    #plot percentile models on top
    z = numpy.linspace(0,zmax,nz)
    if(pcentiles is not None):
        v = numpy.percentile(HH,pcentiles[0],axis=0,method='median_unbiased')        
        s = numpy.zeros((idx,nz))
        ii=0
        for state,x in enumerate(transd_ensemble):
            n = state
            for i,y in enumerate(x):
                d = [0., *y[:n]]
                c = y[n:]
                r,dp = get_sample_is(n+1,d,c,500)
                for j,xx in enumerate(z):
                    s[ii,j] = r[dp.index(next(obj for obj in dp if obj >= xx))]
                ii+=1
        p = numpy.percentile(s,pcentiles[0],axis=0)
        for i,x in enumerate(p):
            ax.plot(x,z,pcentiles[1][i],lw=pcentiles[2][i])

    # plot density of interfaces
    if(interfaceplotcolor is not None):
        X,Y = numpy.meshgrid(numpy.linspace(xmin,xmax,1),numpy.linspace(0,zmax,nz))
        dz=zmax/nz
        den = numpy.zeros(nz)
        for state,x in enumerate(transd_ensemble):
            n = state
            for y in x:
                d = [0., *y[:n]]
                c = y[n:]
                r,dp = get_sample_is(n+1,d,c,500)
                for j in range(1,len(dp)):
                    k = int(dp[j]/dz)
                    if(k< nz): den[k]+=1
        den /= numpy.max(den)
        ax2.fill_betweenx(z, den,color=interfaceplotcolor)

    if(filename is not None): matplotlib.pyplot.savefig(filename)

    return HH

def get_sample_is(n,d,c,zmax):
    #n,msft,d,c=smpl
    depth=[d[0]]
    resistivity=[1/10**c[0]]
    for i in range(1,n):
        depth.append(d[i])
        depth.append(d[i])
        resistivity.append(1/10**c[i-1])
        resistivity.append(1/10**c[i])
    depth.append(zmax*2)
    resistivity.append(1/10**c[-1])
    return resistivity,depth
