from mditre.trainer import parse, Trainer

def main():
	# Parse command line args
	args = parse()

	# random seeds to use for experiments
	seeds = [111, 222, 333, 444, 555, 42, 420, 69, 666, 1437]

	for seed in seeds:
		# reset seed
		args.seed = seed

		# Init trainer object
		trainer = Trainer(args)

		# Load data
		trainer.load_data()

		# run cv loop
		trainer.train_loop()


if __name__ == '__main__':
	main()