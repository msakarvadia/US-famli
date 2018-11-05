import itk
import vtk
import argparse

def main(args):

	reader = vtk.vtkNrrdReader()
	reader.SetFileName(args.img)
	reader.Update()
	img = reader.GetOutput()

	print(img)
	 
	actor = vtk.vtkImageActor()
	actor.GetMapper().SetInputData(img)

	# Setup rendering
	renderer = vtk.vtkRenderer()
	renderer.AddActor(actor)
	renderer.SetBackground(1,1,1)
	renderer.ResetCamera()
	 
	renderWindow = vtk.vtkRenderWindow()
	renderWindow.AddRenderer(renderer)
	 
	renderWindowInteractor = vtk.vtkRenderWindowInteractor()
	renderWindowInteractor.SetInteractorStyle(vtk.vtkInteractorStyleImage())
	renderWindowInteractor.AddObserver("KeyPressEvent", Keypress)
	renderWindowInteractor.SetRenderWindow(renderWindow)
	renderWindowInteractor.Initialize()
	renderWindowInteractor.Start()

def Keypress(obj, event):
    key = obj.GetKeySym()
    if key == "a":
        print("flip x")
    elif key == "s":
        print("flip y")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--img', type=str, help='Input image', required=True)
	parser.add_argument('--transform', type=str, help='Input transformation file', required=True)
	parser.add_argument('--out', type=str, help='Output filename', default="out.nrrd")

	args = parser.parse_args()

	main(args)