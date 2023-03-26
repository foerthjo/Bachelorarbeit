class Score:
	def __init__(this, num_classes, record_summary, class_names):
		this.num_classes = num_classes
		this.true_positives = [0] * num_classes
		this.false_positives = [0] * num_classes
		this.false_negatives = [0] * num_classes
		this.record_summary = record_summary
		this.class_names = class_names
	
	def calculate_ious(this):
		this.ious = [0] * this.num_classes
		mean = 0
		weighted_mean = 0
		total_weight = 0

		for i in range(this.num_classes):
			tp = this.true_positives[i]
			fn = this.false_negatives[i]
			fp = this.false_positives[i]

			gt = tp + fn
			intersection = tp
			union = tp + fp + fn
			
			iou = 0
			if union > 0:
				iou = intersection / union

			mean += iou
			weighted_mean += iou * gt
			total_weight += gt

			this.ious[i] = iou
		
		this.mean = 0
		if this.num_classes > 0:
			this.mean = mean / this.num_classes
		
		this.weighted_mean = 0
		if total_weight > 0:
			this.weighted_mean = weighted_mean / total_weight
	
	def add(this, tp, fp, fn):
		if (this.record_summary):
			this.quick_iou = 0
			for i in range(this.num_classes):
				intersection = tp[i] / this.num_classes
				union = (tp[i] + fp[i] + fn[i])
				if union > 0:
					this.quick_iou += intersection / union

		for i in range(this.num_classes):
			this.true_positives[i] += tp[i]
			this.false_positives[i] += fp[i]
			this.false_negatives[i] += fn[i]

	def quickSummary(this):
		if (this.record_summary):
			return str(this.quick_iou)
		return 'trying to get Score.quickSummary() without record_summary being enabled'

	def __str__(this):
		string = ''
		for i in range(this.num_classes):
			string += str(this.class_names[i]) + ': ' + str(this.ious[i]) + '\n'
		string += 'mean iou: ' + str(this.mean) + '\nweighted mean: ' + str(this.weighted_mean)
		return string