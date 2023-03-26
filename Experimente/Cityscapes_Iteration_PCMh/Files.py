def writeFile(path, content):
	file = open(path, "w")
	file.write(content)
	file.close()