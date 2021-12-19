from mditre.trainer import parse, Trainer

def main():
	# Parse command line args
	args = parse()

	# Init trainer object
	trainer = Trainer(args)

	# Load data
	trainer.load_data()

	# run cv loop
	trainer.train_loop()


if __name__ == '__main__':
	main()