Before Run:
	- If the TA wants to view the results, submitted .ipynb files have results on them or TA can view it on Colab with the Share links below.
	- If the TA wants to run the code, it is suggested that s/he run on the Colab with the Share links below (add shortcut of the Database link to your 	Drive).
	- It should be noted that the databases are hosted by me and have to be accessed by the Share link below.
	- Example runs with outputs and log files are given in the submission.
	- For any problems, please contact Ömer Ünlüsoy, omer.unlusoy@ug.bilkent.edu.tr, +90 542 417 3804

	- Google Colab Link (Style Transfer): https://colab.research.google.com/drive/1qYvlj5EcUXoWtAHq5MraGab7PkSR9YoI?usp=sharing
	- Google Colab Link (for experiment I): https://colab.research.google.com/drive/15Tq7ttrOTahoASyUT21sDlimTx-8mhyM?usp=sharing
	- Google Colab Link (for experiment II): https://colab.research.google.com/drive/1NYRHSfb4BdJFh7CBzkwZzvUrq9IDyOWv?usp=sharing
	- Database Link: https://drive.google.com/drive/folders/1K2mR0r3fJdb3ZVPCKxGJv-9KToaDzo8u?usp=sharing

	- Expected Run Time: < 25 min with Google Colab GPUs

To Run:
	- Style Transfer Colab will ask user if s/he wants to use Van Gogh(V) or Other Artisits(A).
	- For Van Gogh, code expects another input, place; for Other Artists, code expects the name of the Artist from the list.
	- Then, the code starts to run, train.

Experiments:
	- The first experiment is Alpha Ratio (Style Weight / Content Weight).
	- The second experiment is Learning Rate.
	- The third experiment is saving the training procedure as a video which experiments the iteration (epoch) vs output.

Outputs:
	- Final Target Image
	- 3 Images in One Picture (Content, Style, Target)
	- Transformation Video (Training Procedure)

Changes from Computer to Computer:
	- Package Dependencies
	- Google Drive Permissions
	- Datasets from Google Drive (Datasets will be deleted within a month from my personal Drive)
		- Artisits Dataset from Google Drive (these two datasets does not allow access from kaggle and have to be downloaded)
			- can be downloaded from https://www.kaggle.com/ikarus777/best-artworks-of-all-time
		- Van Gogh Dataset from Google Drive
			- can be downloaded from https://www.kaggle.com/ipythonx/van-gogh-paintings
	- Log Filename

Hyperparameters:
	- style_weight (Alpha Ratio)					-> best value: 1e6
	- Learning Rate									-> best value: 0.004
	- steps (iterations)							-> best value: 2100 + 100
	- Content and Style Layers (of the Model)		-> best value: Conv1_1, Conv2_1, Conv3_1, Conv4_1, *Conv4_2, Conv5_1
	- Style Weights									-> best value: 'conv1_1': 1., 'conv2_1': .78, 'conv3_1': .25, 
																	'conv4_1': .22, 'conv5_1': .18

Run Scheme:
	- User can select Van Gogh or all other artists;
		- Van Gogh(V) or All Artists(A):
			- lets's select V.
			- prints all places and styles of Van Gogh:
				Works of the young van Gogh
				Nuenen
				Auvers sur Oise
				Arles
				Sketches in letters
				Villege
				Drawings
				Face
				Saint Remy
				Watercolors
				Paris

				Select type:
					- lets select Arles.
					- random image will be selected in Arles.
					- selected image name: Still Life Vase with Five Sunflowers.jpg
					- training begins with the user's selfie and selected style image.
			
			- or we can select All Artists by typing A.
			- prints all artists:
				Albrecht_Dürer
				William_Turner
				Henri_Rousseau
				Leonardo_da_Vinci
				Rembrandt
				Raphael
				Mikhail_Vrubel
				Hieronymus_Bosch
				Caravaggio
				Edvard_Munch
				Peter_Paul_Rubens
				Joan_Miro
				Pierre-Auguste_Renoir
				Jackson_Pollock
				Pablo_Picasso
				Frida_Kahlo
				Henri_Matisse
				Amedeo_Modigliani
				Gustave_Courbet
				Vincent_van_Gogh
				Gustav_Klimt
				Vasiliy_Kandinskiy
				Diego_Rivera
				Salvador_Dali
				Camille_Pissarro
				Paul_Cezanne
				Kazimir_Malevich
				Henri_de_Toulouse-Lautrec
				Sandro_Botticelli
				Marc_Chagall
				Claude_Monet
				Paul_Gauguin
				Paul_Klee
				Pieter_Bruegel
				Eugene_Delacroix
				El_Greco
				Jan_van_Eyck
				Andy_Warhol
				Giotto_di_Bondone
				Edouard_Manet
				Titian
				Rene_Magritte
				Michelangelo
				Alfred_Sisley
				Andrei_Rublev
				Francisco_Goya
				Georges_Seurat
				Edgar_Degas
				Diego_Velazquez
				Albrecht_Du╠êrer
				Piet_Mondrian

				Select artist:
					- lets select Albrecht_Dürer.
					- random image will be selected in Albrecht_Dürer.
					- selected image name: Albrecht_Dürer_18.jpg
					- training begins with the user's selfie and selected style image.
