using Persa
using DatasetsCF
using Surprise

ds = DatasetsCF.MovieLens()
for noiserate=0.0:0.2:0.9
	print("Noise rate ")
	println(noiserate)
	for base_split=0.1:0.1:0.9
		print("Base Split: ")
		println(base_split)
		holdout = Persa.HoldOut(ds, base_split)
		(ds_train, ds_test) = Persa.get(holdout)
		dataset = copy(ds_train)
	  	if noiserate > 0.0
		    for i=1:size(dataset.file[3])[1]
		    	if rand(1:100) <= noiserate * 100
		    		dataset.file[3][i] = rand(1:5)
		    	end
		    end
	  	end
	  	model = Surprise.KNNBasic(dataset)
	  	Persa.train!(model, dataset)
	  	print(Persa.aval(model, ds_test, Persa.recommendation(ds)))
	  	println("--------------------------------------------------")
	end
end