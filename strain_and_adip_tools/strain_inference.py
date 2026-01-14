# find clusters of pixels that are above a certain threshold
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from PIL import Image
import time
import os
import PIL
from strain_and_adip_tools.schauser_pycellfit import mesh, utils
    
import scipy 
from aggregate_analysis_tools import find_cells_touching_the_border
import importlib
import matplotlib

class StrainInference:
    def __init__(self, img_cell, img_protein, mask_path, protein_sensitivity = 0.5, show_images = False, name = "", img_savepath = "", img_actin = None):
        self.protein_sensitivity = protein_sensitivity
        self.should_show_images = show_images
        self.name = name
        self.img_savepath = img_savepath


        self.background_label = None
        self.array_of_pixels = self.get_array_of_pixels(mask_path)
        # self.hex_mesh = hex_mesh
        self.borders = self.get_borders()

        self.cells_touching_the_border = find_cells_touching_the_border(self.array_of_pixels)
        
        self.img_no_green = img_cell
        self.img_protein = img_protein
        self.img_actin = img_actin

        cell_centers = {}
        # for cell in hex_mesh.cells:
            # cell_centers[cell._label] = cell.approximate_cell_center()
        for i in np.unique(self.array_of_pixels):
            x_bounds = np.argwhere(self.array_of_pixels == i)[:, 0]
            y_bounds = np.argwhere(self.array_of_pixels == i)[:, 1]

            x_b = (x_bounds.min() + x_bounds.max())//2
            y_b = (y_bounds.min() + y_bounds.max())//2

            cell_centers[i] = (y_b, x_b)

        self.all_cell_centers = cell_centers

        assert self.array_of_pixels.shape[:2] == self.img_no_green.shape[:2], f"array of pixels shape: {self.array_of_pixels.shape}, img shape: {self.img_no_green.shape}"
        
        self.mesh = None
        self.hex_mesh = None

        self.all_tensions = None
        self.all_ids = None
        self.all_junction_coords = None

        self.cluster_ids = None
        self.cluster_centers = None
        self.cluster_sizes = None
        self.out_of_bounds = None

        self.tension_arrows = None
        self.tension_positions = None
        self.tension_ids = None


    def get_array_of_pixels(self, mask_path):
        array_of_pixels = np.load(mask_path)
        return array_of_pixels


    def find_clusters(self, green, xlim = None, ylim = None,):
        # normalize the green channel
        green = green - green.min()

        green = green/green.max()

        if xlim is None:
            xlim = (0, green.shape[1])
        if ylim is None:
            ylim = (0, green.shape[0])

        mask = green > self.protein_sensitivity

        mask_no_boundary = np.logical_and(mask, self.borders == 0)

        plt.figure(figsize=(10, 10))

        _, labels = cv2.connectedComponents(mask_no_boundary.astype(np.uint8))

        # find the centers of the clusters that are above a certain size
        min_size = 2
        max_size = 2000
        cluster_centers = []
        cluster_sizes = []
        out_of_bounds = []
        cluster_ids = []


        for i in range(1, labels.max() + 1):
            cluster = labels == i

            if np.sum(cluster) < min_size or np.sum(cluster) > max_size:
                continue

            center = np.mean(np.argwhere(cluster), axis=0)

            if center[1] < xlim[0] or center[1] > xlim[1] or center[0] < ylim[0] or center[0] > ylim[1]:
                out_of_bounds.append(center)
            else:
                cell_id = self.array_of_pixels[int(center[0]), int(center[1])]

                if cell_id == self.background_label:
                    continue

                cluster_centers.append(center[::-1])
                cluster_ids.append(cell_id)

                size_before = np.sum(cluster)

                # center = np.mean(np.argwhere(cluster), axis=0)

                cluster_sizes.append(np.sum(cluster))


        cluster_centers = np.array(cluster_centers)
        cluster_sizes = np.array(cluster_sizes)
        out_of_bounds = np.array(out_of_bounds)
        cluster_ids = np.array(cluster_ids)

        fig, axs = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

        for ax, img in zip(axs, [self.img_no_green, self.array_of_pixels, mask_no_boundary]):
            ax.imshow(img, cmap='Greens')
            ax.imshow(self.img_protein, alpha=0.5, cmap='Greens')

            if len(out_of_bounds.shape) > 1 and out_of_bounds.shape[0] > 0:
                ax.scatter(out_of_bounds[:, 0], out_of_bounds[:, 1], c='b', s=10)

            if len(cluster_centers.shape) > 1 and cluster_centers.shape[0] > 0:
                ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=np.sqrt(cluster_sizes), facecolors='none', edgecolors='r')


        fig.tight_layout(pad = 0.)
        plt.savefig(self.img_savepath + self.name + '_protein_clusters.png', dpi = 300)

        if self.should_show_images:
            plt.show()
        else:
            plt.close()

        assert len(cluster_centers) == len(cluster_sizes)
        
        return cluster_centers, cluster_sizes, cluster_ids, out_of_bounds

    def find_clusters_count_bicellular_junctions(self, green, xlim = None, ylim = None,):

        # normalize the green channel
        green = green - green.min()

        green = green/green.max()

        if xlim is None:
            xlim = (0, green.shape[1])
        if ylim is None:
            ylim = (0, green.shape[0])

        mask = green > self.protein_sensitivity

        # mask_no_boundary = np.logical_and(mask, self.borders == 0)

        plt.figure(figsize=(10, 10))

        _, labels = cv2.connectedComponents(mask.astype(np.uint8))

        # find the centers of the clusters that are above a certain size
        min_size = 2
        max_size = 2000
        cluster_centers = []
        cluster_sizes = []
        out_of_bounds = []
        cluster_ids = []
        on_border = []


        for i in range(1, labels.max() + 1):
            cluster = labels == i

            if np.sum(cluster) < min_size or np.sum(cluster) > max_size:
                continue

            center = np.mean(np.argwhere(cluster), axis=0)

            if center[1] < xlim[0] or center[1] > xlim[1] or center[0] < ylim[0] or center[0] > ylim[1]:
                out_of_bounds.append(center)
            else:
                cell_id = self.array_of_pixels[int(center[0]), int(center[1])]

                if cell_id == self.background_label:
                    continue

                cluster_centers.append(center[::-1])
                cluster_ids.append(cell_id)

                size_before = np.sum(cluster)

                # center = np.mean(np.argwhere(cluster), axis=0)

                cluster_sizes.append(np.sum(cluster))

                if np.sum(np.logical_and(cluster, self.borders)) > 0:
                    on_border.append(True)
                else:
                    on_border.append(False)

        cluster_centers = np.array(cluster_centers)
        cluster_sizes = np.array(cluster_sizes)
        out_of_bounds = np.array(out_of_bounds)
        cluster_ids = np.array(cluster_ids)
        on_border = np.array(on_border)

        fig, axs = plt.subplots(1, 3, figsize=(20, 10), sharey=True)

        for ax, img in zip(axs, [self.img_no_green, self.array_of_pixels, mask]):
            ax.imshow(img, cmap='Greens')
            ax.imshow(self.img_protein, alpha=0.5, cmap='Greens')

            if len(out_of_bounds.shape) > 1 and out_of_bounds.shape[0] > 0:
                ax.scatter(out_of_bounds[:, 0], out_of_bounds[:, 1], c='b', s=10)

            if len(cluster_centers.shape) > 1 and cluster_centers.shape[0] > 0:
                ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=np.sqrt(cluster_sizes), facecolors='none', edgecolors='r')


        fig.tight_layout(pad = 0.)
        try:
            plt.savefig(self.img_savepath + self.name + '_protein_clusters_bicellular_junctions.png', dpi = 300)
        
        except Exception as E:
            print(E)
            print(self.img_savepath)
            print(os.path.isdir(self.img_savepath))
            print("Could not save image!!!")
        if self.should_show_images:
            plt.show()
        else:
            plt.close()

        assert len(cluster_centers) == len(cluster_sizes)
        
        return cluster_centers, cluster_sizes, cluster_ids, out_of_bounds, on_border


    def get_all_ids(self):
        return self.all_ids


    def get_img_from_id(self, id, ax):
        x_bounds = np.argwhere(self.array_of_pixels == id)[:, 0]
        y_bounds = np.argwhere(self.array_of_pixels == id)[:, 1]

        x_b = (x_bounds.min() - 4, x_bounds.max()+4)
        x_b = (max(x_b[0], 0), min(x_b[1], self.img_no_green.shape[0]))

        y_b = (y_bounds.min() - 4, y_bounds.max()+4)
        y_b = (max(y_b[0], 0), min(y_b[1], self.img_no_green.shape[1]))

        _img = self.img_no_green[x_b[0]:x_b[1], y_b[0]:y_b[1]]

        ax.imshow(_img, cmap='gray')

        ax.set_title(f"Cell with id: {id}")

        # ax.imshow(self.array_of_pixels[x_b[0]:x_b[1], y_b[0]:y_b[1]] == id, alpha=0.5)
        ax.imshow(self.borders[x_b[0]:x_b[1], y_b[0]:y_b[1]], alpha=0.5)

        green = self.img_protein
        green = green - green.min()
        green = green/green.max()

        mask = green > self.protein_sensitivity

        overlap = np.logical_and(mask, self.array_of_pixels == id, self.borders == 0)
        
        ax.imshow(overlap[x_b[0]:x_b[1], y_b[0]:y_b[1]], alpha=0.3, cmap='Greens')

        non_counted1 = np.logical_and(mask, self.borders != 0)
        ax.imshow(non_counted1[x_b[0]:x_b[1], y_b[0]:y_b[1]], alpha=0.3, cmap='Reds')

        non_counted2 = np.logical_and(mask, self.array_of_pixels != id)
        ax.imshow(non_counted2[x_b[0]:x_b[1], y_b[0]:y_b[1]], alpha=0.3, cmap='Blues')

        cluster_indexes = np.where(self.cluster_ids == id)[0]

        for cluster_id in cluster_indexes:
            cluster_center = self.cluster_centers[cluster_id]
            cluster_size = self.cluster_sizes[cluster_id]

            zoom_amount_x = self.img_no_green.shape[0]/_img.shape[0]
            zoom_amount_y = self.img_no_green.shape[1]/_img.shape[1]

            zoom_amount = max(zoom_amount_x, zoom_amount_y)*2.

            ax.scatter(cluster_center[0] - y_b[0], cluster_center[1] - x_b[0], s=cluster_size*zoom_amount, facecolors='none', edgecolors='lightblue')



        if id in self.protein_ids:
            protein_arrow_p = self.protein_ids.index(id)
            x = self.protein_arrow_positions[protein_arrow_p][0] - y_b[0]
            y = self.protein_arrow_positions[protein_arrow_p][1] - x_b[0]
            ax.arrow(x, y, self.protein_arrows[protein_arrow_p][0]*10., self.protein_arrows[protein_arrow_p][1]*10., head_width=10, head_length=10, fc='lightblue', ec='lightblue')
        
        if self.tension_ids is not None and id in self.tension_ids:
            tension_arrow_p = self.tension_ids.index(id)
            x = self.tension_positions[tension_arrow_p][0] - y_b[0]
            y = self.tension_positions[tension_arrow_p][1] - x_b[0]
            ax.arrow(x, y, self.tension_arrows[tension_arrow_p][0]*10., self.tension_arrows[tension_arrow_p][1]*10., head_width=10, head_length=10, fc='r', ec='r')
        
        return ax        
    def get_mesh_from_pixel_array(self, array_of_pixels):
        # STEP 2: Generate Mesh 
        hex_mesh = mesh.Mesh(array_of_pixels)
        print()
        hex_mesh.find_cells_from_array()

        self.background_label = hex_mesh.background_label

        print('there are this many cells:')
        print(hex_mesh.number_of_cells)

        hex_mesh.add_edge_points_and_junctions(array_of_pixels)

        print('number of triple junctions:')
        print(hex_mesh.number_of_triple_junctions)

        hex_mesh.make_edges_for_all_cells()
        print('number of edges')
        print(hex_mesh.number_of_edges)

        hex_mesh.generate_mesh()

        # STEP 3: Circle Fit
        hex_mesh.circle_fit_all_edges()

        # # STEP 4: Solve Tensions
        # hex_mesh.solve_tensions()
        # hex_mesh.solve_tensions_new2()
        
        return hex_mesh

    def plot_tensions(self):
        plt.figure(figsize=(13, 13))
        plt.imshow(self.array_of_pixels, cmap='gray', interpolation="nearest", vmax=255)
        for cell in self.hex_mesh.cells:
            cell.plot()
        self.hex_mesh.plot()

        for edge in self.hex_mesh.edges:
            if not edge.outside(self.hex_mesh.background_label):
                # edge.plot_tangent()
                edge.plot(label=True)


        self.hex_mesh.plot_tensions()
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout(pad = 0.)
        plt.savefig(self.img_savepath + self.name + '_tensions.png', dpi = 300)

        if self.should_show_images:
            plt.show()
        else:
            plt.close()


    def find_corresponding_edges(self, id):
        all_edges = list(self.hex_mesh.edges)
        all_edge_labels = [list(edge._cell_label_set) for edge in all_edges]

        edges = []

        for i, edge_labels in enumerate(all_edge_labels):
            if id in edge_labels:
                edges.append(all_edges[i])

        return edges

    def find_corresponding_junctions(self, id):
        all_junctions = list(self.hex_mesh.junctions)
        all_junction_labels = [list(junction.cell_labels) for junction in all_junctions]

        junctions = []

        for i, junction_labels in enumerate(all_junction_labels):
            if id in junction_labels:
                junctions.append(all_junctions[i])

        return junctions

    def find_all_tensions_and_plot(self,):
        array_of_pixels = self.array_of_pixels
        hex_mesh = self.hex_mesh

        fig = plt.figure(figsize=(13, 13))
        plt.imshow(array_of_pixels)

        all_tensions = []
        all_ids = []
        all_junction_coords = []

        for cellid in np.unique(array_of_pixels):
            if cellid == 0:
                continue
                # pass

            edge_tensions = []
            edge_coords = []

            # find the edges that correspond to the cellid
            for cell in hex_mesh.cells:
                if cell._label == cellid:
                    nobreak = False

                    center = cell.approximate_cell_center()
                    plt.plot(center[0],center[1], 'ro', markersize=5)

                    edges = self.find_corresponding_edges(cell._label)

                    for edge in edges:
                        edge_tensions.append(edge.tension_magnitude)
                        edge_coords.append(edge.location)

                        plt.text(edge.location[0], edge.location[1], f'{edge.tension_magnitude*10.:.0f}', fontsize=12, color='white')

                    
            all_tensions.append(np.array(edge_tensions))#/sum(j_tensions)*100)
            all_ids.append(cellid)
            all_junction_coords.append(edge_coords)

        # plt.plot()
        assert len(all_tensions) == len(all_junction_coords)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout(pad = 0.)
        plt.savefig(self.img_savepath + self.name + '_tension_verteces.png', dpi = 300)
        if self.should_show_images:
            plt.show()
        else:
            plt.close()

        return all_tensions, all_ids, all_junction_coords

    def get_arrows_tension(self,):
        if self.all_tensions is None:
            self.set_tension_values()

        plt.figure(figsize=(10, 10))

        plt.imshow(self.img_no_green, cmap='gray')


        tensionarrows = []
        tension_position = []
        tension_ids = []
        for tension, coords, id in zip(self.all_tensions, self.all_junction_coords, self.all_ids):
            if not id in self.all_cell_centers.keys():
                # most likely a background value
                continue

            cell_center = self.all_cell_centers[id]

            if len(coords) == 0:
                continue

            direction = np.array(coords) - np.array(cell_center)
            direction /= np.linalg.norm(direction, axis=1)[:, None]

            tension = np.array(tension)/sum(tension)
            mean_dir = np.average(direction, axis=0, weights=tension)


            # if np.linalg.norm(mean_dir) > 50:
            #     mean_dir = mean_dir/np.linalg.norm(mean_dir)*50

            plt.arrow(cell_center[0], cell_center[1], mean_dir[0]*10., mean_dir[1]*10., head_width=10, head_length=10, fc='r', ec='r')


            tensionarrows.append(mean_dir/np.linalg.norm(mean_dir))
            tension_position.append(cell_center)
            tension_ids.append(id)

            # plt.plot(protpo[0], protpo[1], 'bo', markersize=5)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        # pl        plt.tight_layout(pad = 0.)ot protein clusters

        plt.savefig(self.img_savepath + self.name + '_tension_arrows.png', dpi = 300)

        if self.should_show_images:
            plt.show()
        else:
            plt.close()
        

        self.tension_arrows = tensionarrows
        self.tension_positions = tension_position
        self.tension_ids = tension_ids

        return tensionarrows, tension_position, tension_ids

    def get_arrows_protein(self,):
        if self.cluster_centers is None:
            self.set_cluster_values()

        plt.figure(figsize=(10, 10))
        plt.imshow(self.img_no_green, cmap='gray')

        protein_arrows = []

        protein_arrow_positions = []

        protein_ids = []

        protein_variances = []

        protein_sizes = []  

        unique_ids = np.unique(self.array_of_pixels)

        for id in unique_ids:
            if not id in self.all_cell_centers.keys():
                # most likely a background value
                continue

            if id in self.cells_touching_the_border:
                continue

            cell_center = self.all_cell_centers[id]

            plt.text(cell_center[0], cell_center[1], f'{id}', fontsize=6, color='white')


            protein_centers = self.cluster_centers[self.cluster_ids == id]
            sizes = self.cluster_sizes[self.cluster_ids == id]

            if len(protein_centers) == 0 or len(sizes) == 0:
                protein_arrows.append(np.array([0., 0.]))
                protein_arrow_positions.append(cell_center)
                protein_ids.append(id)
                protein_variances.append(1.)
                protein_sizes.append(0.)
                continue



            dirs_to_protein = protein_centers - cell_center
            norms = np.linalg.norm(dirs_to_protein, axis=1)
            dirs_to_protein /= norms[:, None]

            # EXPERIMENTAL: weigh the directions with the total amount of protein

            # dirs_to_protein = dirs_to_protein * np.max(sizes)/np.max(self.cluster_sizes)

            try:
                average_direction_to_protein = np.average(dirs_to_protein, axis=0, weights=sizes)
                angles = np.arctan2(dirs_to_protein[:, 1], dirs_to_protein[:, 0])
                angles = scipy.stats.circvar(angles)

            except:
                print("It went wrong there")
                print(dirs_to_protein)
                print(sizes)
                continue

            plt.arrow(cell_center[0], cell_center[1], average_direction_to_protein[0]*10., average_direction_to_protein[1]*10., head_width=10, head_length=10, fc='lightblue', ec='lightblue')

            # plt.text(cell_center[0], cell_center[1], f'{sum(self.cluster_ids == id)}', fontsize=12, color='white')
            protein_arrows.append(average_direction_to_protein)
            protein_arrow_positions.append(cell_center)
            protein_ids.append(id)
            protein_variances.append(angles)
            protein_sizes.append(len(sizes))

        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
        plt.tight_layout(pad = 0.)
        plt.savefig(self.img_savepath + self.name + '_protein_arrows.png', dpi = 300)

        if self.should_show_images:
            plt.show()
        else:
            plt.close()


        

        mask = (self.img_protein - self.img_protein.min()) / (self.img_protein.max() - self.img_protein.min()) > self.protein_sensitivity

        mask_no_boundary = np.logical_and(mask, self.borders == 0)
        # plt.imshow(self.img_no_green, alpha =.2, cmap='gray')
        
        from matplotlib.colors import ListedColormap

        plt.figure(figsize=(10, 10))

        cmap_mask = ListedColormap(['none', 'forestgreen'])
        plt.imshow(mask_no_boundary, cmap=cmap_mask, interpolation='none')

        cmap_borders = ListedColormap(['none', 'black'])

        plt.imshow(self.borders, cmap=cmap_borders, interpolation='none')
        # plot_cluster_sizes = self.cluster_sizes*2.

        # alphas = np.sqrt(plot_cluster_sizes/np.max(plot_cluster_sizes))

        

        # edgecolors =  [(0,0,1, 0 if a < 0.15 else 1) for a in alphas]
        # facecolors =  [(0,0,0,0) for _ in alphas]
        # if len(self.cluster_centers.shape) > 1 and self.cluster_centers.shape[0] > 0:
        #     plt.scatter(self.cluster_centers[:, 0], self.cluster_centers[:, 1], s=plot_cluster_sizes, facecolors=facecolors, edgecolors=edgecolors, linewidth=0.75)
        
        for i, arrow in enumerate(protein_arrows):
            plt.arrow(protein_arrow_positions[i][0], protein_arrow_positions[i][1], arrow[0]*10., arrow[1]*10., head_width=10, head_length=10, fc='orangered', ec='firebrick')

        # remove x and y ticks
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')

        plt.tight_layout(pad = 0.)
        plt.savefig(self.img_savepath + self.name + '_protein_clusters_with_borders.png')

        if self.should_show_images:
            plt.show()
        else:
            plt.close()


        # image_np_array = np.ones((self.img_no_green.shape[0], self.img_no_green.shape[1], 4))
        # image_np_array[mask_no_boundary] = [0, 1, 0, 1]
        # image_np_array[self.borders] = [0, 0, 0, 1]
        
        # if self.should_save_images:
        #     plt.savefig(self.img_savepath + self.name + '_protein_clusters_with_borders_no_arrows.png', image_np_array)
        # plt.show()

        # image_np_array = np.ones((self.img_no_green.shape[0], self.img_no_green.shape[1], 4))
        # # image_np_array[mask_no_boundary] = [0, 1, 0, 1]

        # image_np_array[self.borders] = [0, 0, 0, 1]
        # for i, arrow in enumerate(protein_arrows):
        #     plt.arrow(protein_arrow_positions[i][0], protein_arrow_positions[i][1], arrow[0]*10., arrow[1]*10., head_width=10, head_length=10, fc='orangered', ec='firebrick')

        # if self.should_save_images:
        #     plt.savefig(self.img_savepath + self.name + '_no_protein_clusters_with_arrows.png', image_np_array)

        # plt.show()
        protein_magnitudes =  1 - np.array(protein_variances)
        self.protein_arrows = protein_arrows
        self.protein_arrow_positions = protein_arrow_positions
        self.protein_ids = protein_ids

        return protein_arrows, protein_arrow_positions, protein_ids, protein_magnitudes, protein_sizes


    def get_mesh(self,):
        t = time.time()
        hex_mesh = self.get_mesh_from_pixel_array(self.array_of_pixels)
        print(f"Time to get mesh and find tensions: {time.time() - t}")
      
        return hex_mesh

    def setup_hex_mesh(self,):
        if self.hex_mesh is None:
            self.hex_mesh = self.get_mesh()

    def get_borders(self,):
          # find the borders between the cells
        borders = np.zeros(self.array_of_pixels.shape)

        w = 3

        for axis in (0, 1, (0,1)):
            for amount in (-w, w):
                borders = np.logical_or(np.roll(self.array_of_pixels, amount, axis=axis) != self.array_of_pixels, borders)

        # add the edges of the image
        edges = np.zeros(self.array_of_pixels.shape)
        edges[0:(w+1), :] = 1
        edges[-(w+1):-1, :] = 1
        edges[:, 0:(w+1)] = 1
        edges[:, -(w+1):-1] = 1

        borders = np.logical_or(borders, edges)

        if self.should_show_images:
            plt.imshow(borders)
            plt.show()
        return borders
    
    
    
    def set_tension_values(self):
        all_tensions, all_ids, all_junction_coords, = self.find_all_tensions_and_plot()

        self.all_tensions = all_tensions
        self.all_ids = all_ids
        self.all_junction_coords = all_junction_coords

    def set_cluster_values(self,):
        cluster_centers, cluster_sizes, cluster_ids, out_of_bounds = self.find_clusters(self.img_protein)

        self.cluster_centers = cluster_centers
        self.cluster_sizes = cluster_sizes
        self.cluster_ids = cluster_ids
        self.out_of_bounds = out_of_bounds

    def set_cluster_values_count_border(self,):
        cluster_centers, cluster_sizes, cluster_ids, out_of_bounds, on_border = self.find_clusters_count_bicellular_junctions(self.img_protein)

        self.cluster_centers = cluster_centers
        self.cluster_sizes = cluster_sizes
        self.cluster_ids = cluster_ids
        self.out_of_bounds = out_of_bounds

        return on_border



    def do_tension_part(self,):
        self.mesh = self.get_mesh()
        self.plot_tensions()

        t = time.time()
        self.set_tension_values()
        print(f"Time to find tension for each cell: {time.time() - t}")

        t = time.time()
        tension_arrows, tension_positions, tension_ids = self.get_arrows_tension()
        print(f"Time to plot arrows: {time.time() - t}")

        return tension_arrows, tension_positions, tension_ids

    def do_actin_part(self,):
        actin_img = self.img_actin
        
        assert actin_img is not None, "No actin image provided!"

        # normalize the green channel per cell
        actin_img_no_borders = actin_img * (self.borders == 0)

        actin_img_no_borders[actin_img < np.mean(actin_img)] = 0.

        plt.figure(figsize=(10, 10))
        plt.imshow(actin_img_no_borders, cmap='Purples')
        # plt.imshow(self.borders, cmap=matplotlib.colors.ListedColormap(['none', 'black']))
        mean_actin_arrows = []
        actin_positions = []
        actin_ids = []
        actin_variances = []
        # find the centers of the clusters that are above a certain size

        for i in np.unique(self.array_of_pixels):
            if i in self.cells_touching_the_border:
                continue

            xs = np.argwhere(self.array_of_pixels == i)[:, 0]
            ys = np.argwhere(self.array_of_pixels == i)[:, 1]
            
            poss = np.array([ys, xs]).T
            mean = np.mean(poss, axis=0)


            values = actin_img_no_borders[xs, ys]

            if np.sum(values) > 0.:
                weighted_mean = np.average(poss, axis=0, weights=values) - mean
            else:
                weighted_mean = np.array([0., 0.])
            # print(f"Cell id: {i}, weighted mean: {weighted_mean}")
            plt.arrow(mean[0], mean[1], weighted_mean[0], weighted_mean[1], head_width=10, head_length=10, fc='r', ec='r')

            mean_actin_arrows.append(weighted_mean)
            actin_positions.append(mean)
            actin_ids.append(i)

            dirs_to_actin = np.array([ys, xs]).T - mean
            angles = np.arctan2(dirs_to_actin[:, 1], dirs_to_actin[:, 0])
            angles = scipy.stats.circvar(angles)
            actin_variances.append(angles)

        plt.savefig(self.img_savepath + self.name + '_actin_arrows.png', dpi = 300)

        if self.should_show_images:
            plt.show()
        else:
            plt.close()

        actin_magnitudes = 1 - np.array(actin_variances)

        return mean_actin_arrows, actin_positions, actin_ids, actin_magnitudes
    

    def do_corner_analysis(self, count_borders = False):
        self.setup_hex_mesh()
        # self.set_cluster_values()
        if count_borders:
            on_border = self.set_cluster_values_count_border()
        else:
            self.set_cluster_values()
            on_border = None



        # self.cluster_centers = cluster_centers
        # self.cluster_sizes = cluster_sizes
        # self.cluster_ids = cluster_ids
        # self.out_of_bounds = out_of_bounds
        
        
        unique_ids = np.unique(self.array_of_pixels)

        protein_positions = []
        corner_positions = []
        protein_sizes = []
        protein_ids = []
        corner_ids = []
        all_on_border = []

        
        for id in unique_ids:
            if id in self.cells_touching_the_border:
                continue
            # find corners
            juncs = self.find_corresponding_junctions(id)
            for junc in juncs:
                corner_positions.append([junc.x, junc.y])
                corner_ids.append(id)

            # find protein
            protein_centers = self.cluster_centers[self.cluster_ids == id]
            sizes = self.cluster_sizes[self.cluster_ids == id]
            if count_borders:
                on_borders = on_border[self.cluster_ids == id]

            if len(protein_centers) == 0 or len(sizes) == 0:
                continue
            
            for pc in protein_centers:
                protein_positions.append(pc)
                protein_ids.append(id)


            for ps in sizes:
                protein_sizes.append(float(ps))

            if count_borders:
                for ob in on_borders:
                    all_on_border.append(ob)


        plt.imshow(self.borders)
        
        for c in corner_positions:
            plt.scatter(c[0],c[1], c = "r")

        if count_borders:
            cs = ["r" if ob else "b" for ob in all_on_border]
        else:
            cs = ["g" for _ in corner_positions]

        for pc, ps, c in zip(protein_positions, protein_sizes, cs):
            plt.scatter(pc[0], pc[1], c =c, s = ps)

        try:
            plt.savefig(self.img_savepath + self.name + '_protein_clusters_and_junctions.png')
        except:
            print("Could not save image here either!!!")
        if self.should_show_images:
            plt.show()
        else:
            plt.close()

        if not count_borders:
            return protein_positions, protein_ids, protein_sizes, corner_positions, corner_ids, None
        
        return protein_positions, protein_ids, protein_sizes, corner_positions, corner_ids, all_on_border


    def do_protein_part(self,):
        self.set_cluster_values()

        # protein_arrows, protein_arrow_positions, protein_ids, protein_angles, protein_sizes = self.get_arrows_protein()

        return self.get_arrows_protein()

    def make_shuffled_data(self, cell_mask, original_data):
        created_data = np.zeros((original_data.shape[0], original_data.shape[1]), dtype = original_data.dtype)
        # reshuffle thhe pixels within each cell
        for id in np.unique(cell_mask):

            mask = cell_mask == id
            mask = np.where(mask)

            original_data_masked = original_data[mask]
            np.random.shuffle(original_data_masked)
            created_data[mask] = original_data_masked

        # plt.imshow(original_data, cmap='gray')
        # plt.show()
        # plt.imshow(created_data, cmap='gray')
        # plt.show()

        return created_data
    
    
    def do_actin_part_null_model(self, ):
        actin_img = self.make_shuffled_data(self.array_of_pixels, self.img_actin)
        
        assert actin_img is not None, "No actin image provided!"

        # normalize the green channel per cell
        actin_img_no_borders = actin_img * (self.borders == 0)

        actin_img_no_borders[actin_img < np.mean(actin_img)] = 0.

        mean_actin_arrows = []
        actin_positions = []
        actin_ids = []
        actin_variances = []

        for i in np.unique(self.array_of_pixels):
            if i in self.cells_touching_the_border:
                continue

            xs = np.argwhere(self.array_of_pixels == i)[:, 0]
            ys = np.argwhere(self.array_of_pixels == i)[:, 1]
            
            poss = np.array([ys, xs]).T
            mean = np.mean(poss, axis=0)


            values = actin_img_no_borders[xs, ys]

            if np.sum(values) > 0.:
                weighted_mean = np.average(poss, axis=0, weights=values) - mean
            else:
                weighted_mean = np.array([0., 0.])

            mean_actin_arrows.append(weighted_mean)
            actin_positions.append(mean)
            actin_ids.append(i)

            dirs_to_actin = np.array([ys, xs]).T - mean
            angles = np.arctan2(dirs_to_actin[:, 1], dirs_to_actin[:, 0])
            angles = scipy.stats.circvar(angles)
            actin_variances.append(angles)


        actin_magnitudes = 1 - np.array(actin_variances)

        return mean_actin_arrows, actin_positions, actin_ids, actin_magnitudes
    

    def do_multiple_actin_null_models(self, n_repeats = 10):
        all_actin_arrows = []
        all_actin_positions = []
        all_actin_ids = []
        all_actin_magnitudes = []

        for i in range(n_repeats):
            actin_arrows, actin_positions, actin_ids, actin_magnitudes = self.do_actin_part_null_model()
            all_actin_arrows.extend(actin_arrows)
            all_actin_positions.extend(actin_positions)
            all_actin_ids.extend(actin_ids)
            all_actin_magnitudes.extend(actin_magnitudes)

        return all_actin_arrows, all_actin_positions, all_actin_ids, all_actin_magnitudes