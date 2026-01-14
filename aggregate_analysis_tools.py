import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import chisquare
import seaborn as sns

def get_angles(data):
    # get angles of data
    angles = np.arctan2(data[:,1], data[:,0])
    angles = np.rad2deg(angles)
    return angles % 360

def get_resolution(path):
    from PIL.TiffTags import TAGS
    from PIL import Image
    import os
    superpath = "all_data_structured"

    img_path = os.path.join(superpath, path)

    # find a tiff image in the folder
    resolution = -1
    for file in os.listdir(img_path):
        if not file.endswith(".tif") and not file.endswith(".tiff"):
            continue
        with Image.open(os.path.join(img_path, file)) as im:
            meta = {TAGS[key]: value for key, value in im.tag.items()}
            if "XResolution" in meta:
                resolution = meta["XResolution"][0][0] / meta["XResolution"][0][1]
                break

    assert resolution != -1, "No tiff image found in the folder"

    return resolution


def weighted_std(values, weights):
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)*np.sqrt(len(values)/(len(values)-1))




def find_cells_touching_the_border(segmented_cells) -> list:
    """
        returns a list of ids of cells that are touching the border
    """

    border_positions = []
    # find all cells that touching the border
    print(segmented_cells.shape)
    for i in np.unique(segmented_cells):
        if i == 0:  
            continue
        x,y = np.where(segmented_cells == i)
        if (0 in x) or (0 in y) or (segmented_cells.shape[0]-2 in x) or (segmented_cells.shape[1]-2 in y):
            border_positions.append(i)

    return border_positions


def get_arrows_in_stretch(arrow_positions):
    arrows_to_wound = np.zeros((len(arrow_positions), 2))
    arrows_to_wound[:,0] = -1
    arrows_to_wound[:,1] = -1

    norms = np.linalg.norm(arrows_to_wound, axis=1)

    arrows_to_wound_norm = arrows_to_wound/norms[:,None]

    return arrows_to_wound_norm, np.ones(len(arrow_positions), dtype=bool)


def find_closest_point_on_wound(arrow_positions, wound_image):

    wound_positions = np.where((wound_image[:,:,3] == 1.)*(wound_image[:,:,2] == 0.) )
    x,y = wound_positions
    wound_positions = np.array([y, x]).T

    closest_point_on_wound = []

    for a in arrow_positions:
        dist = np.linalg.norm(wound_positions - a[None,:], axis=1)

        closest_point = wound_positions[np.argmin(dist)]
        closest_point_on_wound.append(closest_point)

    return np.array(closest_point_on_wound)


def get_arrows_to_wound(arrow_positions, wound_image):
    # get the closest point on the wound for each arrow position
    closest_point_on_wound = find_closest_point_on_wound(arrow_positions, wound_image)

    # get the direction of the arrow to the wound
    arrows_to_wound = closest_point_on_wound - arrow_positions

    # normalize the arrows to wound
    # norms = np.linalg.norm(arrows_to_wound, axis=1)
    arrows_to_wound_norm = arrows_to_wound#/norms[:,None]

    # check if the arrow is in the wound
    # usable = norms > 0.5

    return arrows_to_wound_norm#, usable


def get_angles_to_wound(fn, should_stretch = False, path = ""):

    wound_str = path + 'wounds/'+fn+'_WOUND.png'

    has_wound = os.path.isfile(wound_str)
    

    arrow_positions = np.loadtxt(path + 'outputs/protein/protein_positions_'+fn+'.csv', delimiter=',')
    arrows = np.loadtxt(path + 'outputs/protein/protein_arrows_'+fn+'.csv', delimiter=',')
    arrows[:,1] = -arrows[:,1]
    arrow_ids = np.loadtxt(path + 'outputs/protein/protein_ids_'+fn+'.csv', delimiter=',')
    segmented_cells = np.load(path + '/masks/mask_'+fn+'.npy')

    if has_wound:
        print("has wound!")
        wound = plt.imread(wound_str)
        arrows_to_wound, usable = get_arrows_to_wound(arrow_positions, wound)
    elif should_stretch:
        print("does not have wound, but stretch!")
        arrows_to_wound, usable = get_arrows_in_stretch(arrow_positions)
    else:
        print("does not have wound!")
        # x unit vector
        # arrows_to_wound = np.zeros((len(arrow_positions), 2))
        # arrows_to_wound[:,0] = 1
        if not "Fig-7" in fn:
            random_dir = np.random.rand() * 2*np.pi
            arrows_to_wound = np.array([np.cos(random_dir), np.sin(random_dir)])
            arrows_to_wound = np.tile(arrows_to_wound, (len(arrow_positions), 1))
        else:
            # point downwards
            arrows_to_wound = np.zeros((len(arrow_positions), 2))
            arrows_to_wound[:,1] = -1
        usable = np.ones(len(arrows), dtype=bool)



    cells_touching_border = find_cells_touching_the_border(segmented_cells)

    for i, id in enumerate(arrow_ids):
        if id in cells_touching_border:
            usable[i] = False

    if 0 in arrow_ids:
        indx = np.where(arrow_ids == 0)[0]
        usable[indx] = False
        

    angles_to_wound = get_angles(arrows_to_wound)
    angles = get_angles(arrows)

    # get the angle between the two

    angles = angles[usable]
    angles_to_wound = angles_to_wound[usable]
    arrows_to_wound = arrows_to_wound[usable]

    angle_diff = angles - angles_to_wound
    angle_diff = angle_diff % 360


    poss = arrow_positions[usable]
    fig, ax = plt.subplots(1,1,figsize=(10,10))

    plt.title(fn)
    ax.imshow(segmented_cells, cmap='tab20')
    if has_wound:
        ax.imshow(wound)

        ax.quiver(poss[:,0], poss[:,1], arrows_to_wound[:,0], arrows_to_wound[:,1], color='red')
    else:
        ax.imshow(segmented_cells, cmap='tab20')
        
    norm_arrows = arrows[usable]/np.linalg.norm(arrows[usable], axis=1)[:,None]
    
    ax.quiver(poss[:,0], poss[:,1], norm_arrows[:,0], norm_arrows[:,1], color = "white",)

    for i in np.unique(segmented_cells):
        if i == 0:
            continue
        x,y = np.where(segmented_cells == i)
        ax.text(np.mean(y), np.mean(x), str(i), color='white', fontsize=8, ha='center', va='center')

    plt.show()

    fig, ax = plt.subplots(1,1,figsize=(10,10))


    # _img = plt.imread(path + 'images/'+fn+'_protein_clusters_with_borders_no_arrows.png')
    _img = np.load(path + 'images/'+fn+'_protein_clusters_with_borders_no_arrows.npy')

    # make the green pixels darker
    _img[_img[:,:,1] > _img[:,:,0] ] = _img[_img[:,:,1] > _img[:,:,0]]*0.5
    _img[:,:,3] = 1.


    plt.imshow(_img)
    plt.axis('off')

    def make_pretty_arrows(arrows, positions, color, s = 10):
        for p, a in zip(positions, arrows):
            plt.arrow(p[0], p[1], a[0], -a[1], facecolor=color, edgecolor = "dark"+color, head_width=s/2., head_length=s*2)

    if has_wound:
        ax.imshow(wound, alpha = 0.5)

    make_pretty_arrows(arrows_to_wound, poss, 'blue', s= 5)
    make_pretty_arrows(norm_arrows, poss, 'red')
    
    for p in arrow_positions[~usable]:
        plt.plot(p[0], p[1], 'ro')

    plt.savefig(path + 'images/4paper/'+fn+'_protein_clusters_with_borders_and_arrows.png')
    plt.show()
    fig, ax = plt.subplots(1,1,figsize=(10,10))


    _img = np.load(path + 'images/'+fn+'_protein_clusters_with_borders_no_arrows.npy')

    # remove all green pixels
    _img[_img[:,:,1] > _img[:,:,0] ] = 1

    plt.imshow(_img)
    plt.axis('off')
    if has_wound:
        ax.imshow(wound, alpha = 0.5)
    
    for p in arrow_positions[~usable]:
        plt.plot(p[0], p[1], 'ro')

    make_pretty_arrows(norm_arrows, poss, 'red')
    plt.savefig(path + 'images/4paper/'+fn+'no_protein_clusters_with_borders.png')
    plt.show()

    return angle_diff, usable


import networkx as nx

def get_shortest_distances(mask, wound):
    
    segmented_cells = mask
    wound_mask = (wound[:,:,3] == 1.)*(wound[:,:,2] == 0.) 
    
    wound_positions = np.where(wound_mask)
    wound_positions = np.array([wound_positions[1], wound_positions[0]]).T

 

    print("wound_positions", wound_positions.shape)
    print("segmented_cells", segmented_cells.shape)
    # add the wound to the segmented cells
    wound_number = np.unique(segmented_cells).shape[0]

    segmented_cells[wound_positions[:,1], wound_positions[:,0]] = wound_number



    # check which cells border:
    # ud = np.roll(segmented_cells, 5, axis=0)
    # lr = np.roll(segmented_cells, 5, axis=1)

    ud = segmented_cells[:-5, :]
    lr = segmented_cells[:, :-5]
    ud = np.concatenate((segmented_cells[-5:, :], ud), axis=0)
    lr = np.concatenate((segmented_cells[:, -5:], lr), axis=1)

    # discar the roll

    N = np.max(np.unique(segmented_cells))+1
    nb_matrix = np.zeros((N, N))



    for type in np.unique(segmented_cells):
        if type == wound_number:
            continue
        if type == 0:
            continue

        where_type = np.where(segmented_cells == type)

        for type_position in np.array([where_type[1], where_type[0]]).T:
            yy, xx = type_position

            if xx <= 6 or yy <= 6 or xx >= segmented_cells.shape[0]-6 or yy >= segmented_cells.shape[1]-6:
                continue
            
            if ud[xx, yy] != 0 and segmented_cells[xx, yy] != ud[xx, yy]:
                
                try:
                    nb_matrix[segmented_cells[xx, yy], ud[xx, yy]] = 1
                    nb_matrix[ud[xx, yy], segmented_cells[xx, yy]] = 1
                except Exception as e:
                    print(e)
                    print(xx,yy, segmented_cells.shape, )
                    print(segmented_cells[xx, yy], ud[xx, yy])
                    print(nb_matrix.shape)
            if  lr[xx, yy] != 0 and segmented_cells[xx, yy] != lr[xx, yy]:
                nb_matrix[segmented_cells[xx, yy], lr[xx, yy]] = 1
                nb_matrix[lr[xx, yy], segmented_cells[xx, yy]] = 1
    # make symmetric


    nb_matrix = np.maximum(nb_matrix, nb_matrix.T)



    # plt.imshow(segmented_cells, cmap='tab20')
    # plt.show()


    G = nx.from_numpy_array(nb_matrix)
    # for each cell, get the shortest path to the wound
    shortest_paths = {}
    for type in np.unique(segmented_cells):
        if type == wound_number:
            continue
        try:
            shortest_path = len(nx.shortest_path(G, source=type, target=wound_number))-1
        except nx.NetworkXNoPath:
            shortest_path = 999

        shortest_paths[type] = shortest_path

    return shortest_paths

def make_distance_to_wound_plot(shortest_paths, abs_angle_diff, title):
    fig, ax = plt.subplots(figsize=(8,6))

    # plt.errorbar(x, y, yerr=yerr, fmt='o', color='darkblue', capsize=3)
    
    sns.swarmplot(x=shortest_paths, y=abs_angle_diff, color='darkblue', ax=ax)
    # boxplot
    sns.boxplot(x=shortest_paths, y=abs_angle_diff, color='white', ax=ax, linewidth=1.5, fliersize=0)

    # plot the averages
    x = np.unique(shortest_paths)
    # remove the -1
    x = x[x != -1]
    y = []
    yerr = []
    for b in x:
        # weights = lengths[usable][shortest_paths == b]
        # weights = np.ones_like(abs_angle_diff[shortest_paths == b])
        y.append(np.median(abs_angle_diff[shortest_paths == b]))
        # yerr.append(weighted_std(abs_angle_diff[shortest_paths == b], weights=weights))

    print(x)
    plus = 1 if -1 in np.unique(shortest_paths) else 0
    xx = [i+plus for i in range(len(x))]
    plt.plot(xx, y, color='red', marker='o', linestyle='-', )        
    ax.set_xlabel('Distance to wound in cells')
    ax.set_ylabel('Angle difference to wound')
    plt.title(title)
    plt.show()




def make_rose_plot(arrows, protein_sizes, protein_sigmas, exclude_empty = False, ax = None, nbins = 12, N_no_angle = 10, title = "", fit = False, resolution = 1., relative = 0., other = None, plotother = False,):
    cutoff = 0 if exclude_empty else -1

    angles = np.deg2rad(get_angles(arrows))[protein_sizes > 0]
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    

    angles -= relative

    angles = (angles + np.pi) % (2*np.pi)

    # rotate all angles by 15 degrees
    angles = (angles + np.pi/12) % (2*np.pi)

    # ax.hist(np.deg2rad(angles), bins=23, color='b', alpha=0.7, histtype='step')
    density = True
    hist, bins = np.histogram(angles, bins=nbins, density=density, range = (0, 2*np.pi))
    ahist, _ = np.histogram(angles, bins=nbins, density=False, range = (0, 2*np.pi))

    non_density_hist = np.histogram(angles, bins=nbins, density=False, range = (0, 2*np.pi))[0]


    if other is not None:
        other_angles = np.deg2rad(get_angles(other))
        other_angles -= relative
        other_angles = (other_angles + np.pi) % (2*np.pi)
        other_angles = (other_angles + np.pi/12) % (2*np.pi)

        other_non_density_hist = np.histogram(other_angles, bins=nbins, density=False, range = (0, 2*np.pi))[0]
        print("other_non_density_hist", other_non_density_hist)
        print(len(angles), len(other_angles))
        other_non_density_hist = other_non_density_hist*float(len(angles)/len(other_angles))

        other_hist, _ = np.histogram(other_angles, bins=nbins, density=density, range = (0, 2*np.pi))


    dd = np.mean(ahist[hist!=0]/hist[hist!=0])

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        N = len(angles)
        area = hist / N
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5

        if other is not None:
            other_radius = (other_hist/N/np.pi) ** .5
    # Otherwise plot frequency proportional to radius
    else:
        radius = hist

    # move the bins back to the original position
    bins = bins - np.pi/12
    
    if not plotother:
        bars = ax.bar((bins[:-1] + bins[1:])/2, radius, width=np.diff(bins), align="center", edgecolor='darkblue', color='deepskyblue', fill=True, alpha=1., linewidth=1.5)

  

    ax.set_theta_zero_location('E')
    ax.set_theta_direction(-1)


    ax.set_xticklabels([])

    yt = np.linspace(0, np.max(radius), 6)
    
    ax.set_yticks(yt)
    if density:
        ytlbs = [f"{int(np.ceil(y**2*np.pi*N*dd))}" for y in yt]
        print(ytlbs)
        ax.set_yticklabels(ytlbs)
    else:
        ax.set_yticklabels([f"{int(y)}" for y in yt])


    ax.set_rlabel_position(180)
    ax.set_xticks(np.linspace(0, 2*np.pi, 9), ['180', '-135', '-90', '-45', '0', '45', '90', '135', ' '])

    extratitle = f"\n{len(angles)} cells | 1-Var: Mean {np.mean(protein_sigmas[protein_sizes>cutoff]):.3} $\pm$ {np.std(protein_sigmas[protein_sizes>cutoff]):.3}"
    
    if fit:
        # draw the mean angle
        mean_radius = np.mean(radius)
        tt = np.linspace(0, 2*np.pi, 50)
        r = np.ones_like(tt)*mean_radius
        ax.plot(tt, r, color='red', linestyle='--')

        cs = chisquare(non_density_hist)
        print(f"p-value: {cs.pvalue} | chi2: {cs.statistic} | N: {len(non_density_hist)}")
        print(non_density_hist)
        extratitle += f"\n$\chi^2$ p-value: {cs.pvalue:.3} | $\chi^2$: {cs.statistic:.3}"


    if other is not None:
        if plotother:
            # ax.bar((bins[:-1] + bins[1:])/2, other_radius, width=np.diff(bins), align="center", edgecolor='red', fill=True, alpha=0.2)
            # ax.bar((bins[:-1] + bins[1:])/2, other_radius, width=np.diff(bins), align="center", edgecolor='red', fill=False, alpha=1.)
            scls = 1/(other_radius/np.mean(radius))

            ax.bar((bins[:-1] + bins[1:])/2, scls*radius, width=np.diff(bins), align="center", edgecolor='green', fill=True, alpha=0.2)
            ax.bar((bins[:-1] + bins[1:])/2, scls*radius, width=np.diff(bins), align="center", edgecolor='green', fill=False, alpha=1.)
            print(np.sum(scls*radius), np.sum(radius) )
                

        else:
            mean_radius = np.mean(radius)
            tt = np.linspace(0, 2*np.pi, 50)
            r = np.ones_like(tt)*mean_radius
            ax.plot(tt, r, color='red', linestyle='--')

        cs = chisquare(non_density_hist, other_non_density_hist)

        print(f"p-value: {cs.pvalue} | chi2: {cs.statistic} | N: {len(non_density_hist)}")
        print(non_density_hist)
        extratitle += f"\n$NULL MODEL \chi^2$ p-value: {cs.pvalue:.3} | $\chi^2$: {cs.statistic:.3}"

    if exclude_empty:
        extratitle += f"\nExcluding {sum(protein_sizes<=cutoff)} empty cells"

    ax.set_title(title + extratitle, pad = 36 if fit else 24)
    return ax



def add_files_to_dict(files, dat, active_path, quantities):
    for f in files:
        fname = f
        frame = -1
        isvideo = "frame" in f

        if not isvideo:
            fname = "_".join(f.split("_")[2:])[:-4]
        else:
            frame = int(f.split("frame_")[1].split("_")[0])
            fname = "_".join(f.split("_")[4:])[:-4]

            for q in quantities:
                if not fname in dat[q]:
                    dat[q][fname] = {}

        for q in quantities:
            if q in f:
                if not isvideo:
                    dat[q][fname] = np.loadtxt(active_path + "/" + f, delimiter=",")
                else:
                    dat[q][fname][frame] = np.loadtxt(active_path + "/" + f, delimiter=",")


def import_data(subpath, interest, subfolders = False):
    superpath = "all_data_structured"


    path = os.path.join(superpath, subpath)


    assert interest in ["corner", "protein", "actin", "actin_null"], "interest must be either corner, protein or actin"

    all_quantitites = {"protein": ["protein_arrows", "protein_ids", "protein_positions", "protein_sizes", "protein_magnitudes", "protein_sizes"],
                 "corner": ["corner_ids", "corner_positions", "protein_ids", "protein_positions", "protein_sizes", "on_border"],
                 "actin" : ["actin_arrows", "actin_ids", "actin_positions","actin_magnitudes"],
                 "actin_null" : ["null_arrows", "null_ids", "null_positions","null_magnitudes"]}


    quantities = all_quantitites[interest]


    data = {}

    if subfolders:
        folders = [p for p in os.listdir(path) if os.path.isdir(os.path.join(path, p))]
        for p in folders:
            folderpath = os.path.join(path, p)
            active_path = os.path.join(folderpath, "results", interest)

            print("Looking at folder:")
            print(folderpath)
            files = os.listdir(active_path)

            data[p] = {}
            for q in quantities:
                data[p][q] = {}

            add_files_to_dict(files, data[p], active_path, quantities)
    else:
        active_path = os.path.join(path, "results", interest)
        # Fallback: if nested 'results/results', try direct interest folder
        if not os.path.isdir(active_path):
            alt = os.path.join(path, interest)
            if os.path.isdir(alt):
                active_path = alt
            else:
                raise FileNotFoundError(f"Directory not found: {active_path}")
        print("Looking at folder:")
        print(active_path)
        files = os.listdir(active_path)

        for q in quantities:
            data[q] = {}


        add_files_to_dict(files, data, active_path, quantities)
     
    return data