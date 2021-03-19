# 3D_ConvexHull-CollisionDetect-

#Requirments:
	pip install moderngl
	pip install objloader
#Use:
	  the def:GenerateAHull(choice,vertnum)  will produce a polyhedron.
		when chioce is in 1~5 ,the hull will build from a series of default point set.If choice is another number,the hull will build from a random point set,The set will have vertnum points. 
	the def:CollisionDetection(choice,vertnum) will produce two polyhedron and detect the conllision.
		When choice is in 1~4,the program will produce two hull from two default sets.If choice another number,they code will produce two random polyhedron and do detection.
		If the two polyhedrons have coliision,they will be rendering in red color,otherwise in blue .
