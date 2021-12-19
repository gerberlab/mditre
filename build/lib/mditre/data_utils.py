import pickle, warnings, json, itertools, re, copy

try:
    import configparser as ConfigParser
except:
    from six.moves import configparser as ConfigParser

import numpy as np
from ete3 import Tree
import pandas as pd


### 16s and Metaphlan data loading adapted from MITRE code ###
dequote = lambda s: s.strip('"')


def select_variables(dataset, keep_variable_indices):
    """ Copy the dataset, retaining only specified 
    variables. 

    Raises ValueError if keep_variable_indices is empty.

    Note that, if dataset has a variable_tree attribute, 
    the tree will be pruned to keep only those nodes which are
    kept variables, and the additional nodes required to preserve the
    topology of the tree connecting them; thus, not all nodes in the 
    resulting variable_tree are guaranteed to be variables. 

    """
    if not keep_variable_indices:
       raise ValueError('No variables to be kept.')

    new_variable_names = [dataset.variable_names[i] for
                          i in keep_variable_indices]
    # We want to index into copies of the arrays in dataset.X
    # so that the underlying data is copied instead of referenced.
    temp_dataset = dataset.copy()
    new_X = []
    for subject_X in temp_dataset.X:
        if len(subject_X) == 0:
            new_X.append(subject_X)
        else:
            new_X.append(subject_X[keep_variable_indices])
    new_variable_weights = (
        temp_dataset.variable_weights[keep_variable_indices]
    )
    # This seems a bit redundant but leaves open the possibility
    # of changes to the internals of the Dataset class, 
    # ensures that the n_variables attribute is updated, etc.
    # TODO: make this a copy operation and then update only
    # the needed attributes...
    new_dataset = MITREDataset(
        new_X, temp_dataset.T, temp_dataset.y,
        new_variable_names, new_variable_weights,
        temp_dataset.experiment_start,
        temp_dataset.experiment_end,
        subject_IDs=temp_dataset.subject_IDs,
        subject_data=temp_dataset.subject_data,
        additional_subject_categorical_covariates=temp_dataset.additional_subject_categorical_covariates,
        additional_covariate_default_states=temp_dataset.additional_covariate_default_states,
        additional_subject_continuous_covariates=temp_dataset.additional_subject_continuous_covariates,
        variable_annotations = temp_dataset.variable_annotations.copy()
    )
    if hasattr(temp_dataset, 'variable_tree'):
        new_tree = temp_dataset.variable_tree.copy()
        old_node_names = {n.name for n in new_tree.get_descendants()}
        new_nodes = [v for v in new_variable_names if v in old_node_names]
#        print 'debug select variables: new nodes:'
#        print new_nodes
        if new_nodes:
            new_tree.prune(new_nodes, preserve_branch_length=True)
            new_dataset.variable_tree = new_tree
        # Otherwise there is no point in retaining the tree as we
        # have dropped all variables with a tree relationship.
    return new_dataset

def select_subjects(dataset, keep_subject_indices, invert=False):
    """ Copy the dataset, retaining only specified 
    subjects. 
    
    Raises ValueError if keep_subject_indices is empty.

    If invert is True, keep all subjects _except_ those specified.

    dataset - rules.Dataset instance
    keep_subject_indices - list or array of numbers, the indices (into
    dataset.X/dataset.T/dataset.subject_IDs) of the subjects to be
    retained.

    """
    if len(keep_subject_indices) < 1:
       raise ValueError('No subjects to be kept.')

    if invert:
        exclude_indices = set(keep_subject_indices)
        keep_subject_indices = [i for i in range(dataset.n_subjects) if
                                i not in exclude_indices]
    new_data = dataset.copy()
    if new_data.additional_covariate_matrix is not None:
        new_data.additional_covariate_matrix = dataset.additional_covariate_matrix[keep_subject_indices]
    new_X = []
    new_T = []
    new_y = []
    new_subject_IDs = []
    for i in keep_subject_indices:
        new_X.append(new_data.X[i])
        new_T.append(new_data.T[i])
        new_y.append(new_data.y[i])
        new_subject_IDs.append(new_data.subject_IDs[i])
    new_data.X = new_X
    new_data.T = new_T
    new_data.y = np.array(new_y)
    new_data.subject_IDs = new_subject_IDs
    new_data.n_subjects = len(new_subject_IDs)
    if isinstance(new_data.subject_data, pd.DataFrame):
        new_data.subject_data = new_data.subject_data.loc[new_subject_IDs]
    new_data._primitive_result_cache = {}
    return new_data


def discard_low_overall_abundance(data, min_abundance_threshold,
                                  skip_variables=set()):
    """ Drop taxa whose summed data over all observations is too low.

    Specifically, for each variable (except those given by name in
    skip_variables), add together the data for each observation 
    in each subjects. Discard the variable if this is less than 
    min_abundance_threshold.

    Returns: a new Dataset object and an array of indices of variables
    _kept_ in the filtering process (to allow the same transformation to
    be performed on other data.) 

    A ValueError will be raised if the selected conditions filter out
    all the variables.

    """
    keep_indices = []
    summed_data = np.zeros(data.n_variables)
    for subject_data in data.X:
        # recall timepoints are the columns of entries of X
        for timepoint in subject_data.T:
            summed_data += timepoint

    passing_variables = summed_data >= min_abundance_threshold
    for i,passes in enumerate(passing_variables):
        if passes or (data.variable_names[i] in skip_variables):
            keep_indices.append(i)

    return select_variables(data, keep_indices), keep_indices


def discard_low_depth_samples(data, min_abundance_threshold):
    """ Drop observations where data summed over OTUs is too low.

    For example, we apply this when some samples had too low 
    sequencing depth (or at least too few sequences which survived
    the pipeline.)
    
    For each subject, the data is added together along the OTU axis,
    and only those timepoints where the result is greater 
    than threshold are kept.

    Returns: a new Dataset object.

    """
    new_data = data.copy()
    for i in range(new_data.n_subjects):
        table = new_data.X[i]
        if len(table) == 0:
            # Subjects with no observations are okay.
            continue
        depths = np.sum(table,axis=0)
        keep_indices = (depths >= min_abundance_threshold)
        table = table.T[keep_indices].T
        new_data.X[i] = table
        new_data.T[i] = new_data.T[i][keep_indices]
    return new_data

def trim(data, t0, t1):
    """ Drop observations before t0 or after t1.

    Returns a new Dataset object, with experiment_start set to t0
    and experiment_end set to t1.

    """
    new_data = data.copy()
    for i in range(new_data.n_subjects):
        times = new_data.T[i]
        table = new_data.X[i]
        keep_indices = (times >= t0) & (times <= t1)
        table = table.T[keep_indices].T
        new_data.X[i] = table
        new_data.T[i] = new_data.T[i][keep_indices]
    new_data.experiment_start = t0
    new_data.experiment_end = t1
    return new_data

def test_timepoints_0(timepoints, n_samples, n_intervals,
                      start, end, n_consecutive=1):
    # Likely a more efficient way to do this exists...
    boundaries = np.linspace(start, end, n_intervals+1)
    for i,t0 in enumerate(boundaries[:-n_consecutive]):
        t1=boundaries[i+n_consecutive]
        n_timepoints_in_window = np.sum(
            (t0 <= timepoints) &
            (timepoints <= t1)
        )
        print('%f %f %d' % (t0, t1, n_timepoints_in_window))
        if n_timepoints_in_window < n_samples:
            return False
    return True

def filter_on_sample_density(data, n_samples, interval,
                             method=0, n_consecutive=1):
    """Discard subjects with insufficient temporal sampling.

    In general, any subject without n_samples observations 
    in certain time periods of length controlled by the 
    'interval' argument will be dropped. 

    If method=0, currently the only choice, divide the window
    [data.experiment_start, data.experiment_stop] into 'interval'
    equal pieces. If, in any continuous block of 
    n_consecutive such pieces, there are fewer than
    n_samples observations, drop the subject.

    Returns a new dataset. n_subjects, subject_IDs, and subject_data
    will be updated appropriately (assuming subject_data is a 
    pandas DataFrame: if not, it is left alone.)

    """
    keep_indices = []
    for i, timepoints in enumerate(data.T):
        if method==0:
            test = test_timepoints_0
        else:
            raise ValueError
        okay = test(timepoints, n_samples, interval,
                    data.experiment_start,
                    data.experiment_end, n_consecutive=n_consecutive)
        if okay:
            print('passing %d:' % i)
            print(timepoints)
            keep_indices.append(i)
        else:
            print('failing %d:' % i)
            print(timepoints)
            
    return select_subjects(data,keep_indices)

def discard_where_data_missing(data, field):
    """ Discard subjects where data for a particular field is missing. 

    Assumes the missing data value is NaN. Non-numeric values
    are never considered missing, even the empty string.

    """
    keep_indices = []
    for i, value in enumerate(data.subject_data[field].values):
        if not (np.isreal(value) and np.isnan(value)):
            keep_indices.append(i)
    return select_subjects(data, keep_indices)


class MITREDataset:
    def __init__(self, X, T, y, 
                 variable_names, variable_weights,
                 experiment_start, experiment_end,
                 subject_IDs=None, subject_data=None,
                 additional_subject_categorical_covariates=[],
                 additional_covariate_default_states=[],
                 additional_subject_continuous_covariates=[],
                 variable_annotations={}):
        """ Store experimental data in an object.

        We assume that most preprocessing (variable selection, creation
        of aggregate data for higher taxa, rescaling and renormalization,
        etc.) has been done already, and that some prior on the 
        likelihood of rules in the model applying to each variable
        has already been calculated. 

        Arguments:

        X - list, containing for each subject an n_variables by 
        n_timepoints_for_this_subject array of observations.
        T - list, containing for each subject a vector giving the
        observation timepoints (note the length will be different for
        each subject in general.)
        y - list/array of boolean or 0/1 values indicating whether each 
        subject developed the condition of interest. 
        variable_names - list of strings, used for formatting output
        variable_weights - array of weights for each variable which will
        be used later to calculate the prior probability that a rule applies
        to that variable. 
        experiment_start - Time at which the experiment started.
        experiment_end - Time at which the experiment ended. 
        subject_IDs - list of identifiers for each experimental subject
        (currently purely for reference)
        subject_data - Optional pandas dataframe giving additional information 
        about each subject. 
        additional_subject_categorical_covariates - list, optional (see below)

        additional_covariate_default_states - list, optional. If these
        two arguments are given, additional covariates are included in
        the logistic regression, which do not depend on the rule
        set. The covariates should be the names of columns in the
        dataframe passed as subject_data. For each column, a list of
        unique values is generated.  For each unique value OTHER than
        the corresponding entry in additional_covariate_default_states,
        a new covariate is generated, which is 1 for subjects for whom
        the feature takes that value, and 0 otherwise (effectively, a
        one-hot encoding leaving out the default value.) The matrix of 
        covariates is stored in self.additional_covariate_matrix, and a
        list of (feature, value) pairs corresponding to each column is 
        stored in self.additional_covariate_encoding.         

        additional_subject_continuous_covariates - list, optional. If
        this argument is given, additional covariates are included in
        the logistic regression, which do not depend on the rule set.
        The covariates should be the names of columns in the dataframe
        passed as subject_data. For each column, a list of unique
        values is generated. Unlike the categorical covariates, no
        default state is specified and no encoding is performed-
        instead, each specified column in the dataframe is adjusted to
        have zero mean and then inserted directly into the regression
        covariate matrix. (Note that no rescaling is done, and the
        mean-centering behavior is only a convenience- fundamentally,
        it is the user's responsibility to transform the data as they
        feel is appropriate before incorporating it into the
        regression! In particular, the user should consider the scale
        of this covariate relative to the columns in the covariate
        matrix which come from the rule set or the discrete
        covariates- all entries in those columns will be either 0 or
        1.) Note that these values _should never be NaN_. This is not
        explicitly checked for at this stage for consistency with the
        behavior of the main.py script for categorical variables:
        subjects with NaNs in relevant columns are dropped _after_
        initial generation of the dataset and then the covariate
        matrix is regenerated. A later release may revisit this
        decision.

        variable_annotations: dict mapping variable names to taxonomic or 
        other descriptions, optional

        This method sets up the following useful attributes:
        self.n_subjects
        self.n_variables
        self.n_fixed_covariates (equal to 1 (for the constant term) + the number
        of columns in self.additional_covariate_matrix)

        Dataset objects also offer convenience methods
        apply_primitive, apply_rules, and stratify, which determine
        the output of primitives, rules, or rule lists applied to the
        data. This allows for various caching approaches that speed up
        these evaluations: currently implemented as an attribute
        _primitive_result_cache, a dict (empty by default) of arrays
        of booleans giving the truths of primitives, expressed as
        tuples, for each subject in the dataset.

        Raises ValueError if the number of variables reported
        on in each observation table in X differs from the number
        of variable names provided, or if that number does not
        match the dimension of the argument variable_weights.

        """
        self.X = X 
        self.T = T 
        self.y = np.array(y,dtype='bool')
        self.variable_names = variable_names
        self.variable_weights = variable_weights
        self.experiment_start = experiment_start
        self.experiment_end = experiment_end
        self.subject_data = subject_data
        self.subject_IDs = subject_IDs
        self.n_subjects = len(X)
        self.n_variables = len(variable_weights)
        self._primitive_result_cache = {}
        for array in X:
            # Allow a special case where a subject has no
            # observations
            if len(array) == 0:
                continue
            this_subject_n_variables, _ = array.shape
            if this_subject_n_variables != self.n_variables:
                raise ValueError('Observation-prior dimension mismatch.')
        if len(self.variable_names) != self.n_variables:
            raise ValueError('Incorrect number of variable names.')

        self.additional_subject_categorical_covariates = additional_subject_categorical_covariates
        self.additional_covariate_default_states = additional_covariate_default_states
        self.additional_subject_continuous_covariates = additional_subject_continuous_covariates
        self.generate_additional_covariate_matrix()
        self.variable_annotations = variable_annotations

    def generate_additional_covariate_matrix(self):
        """ Encode additional covariate matrix at initialization.

        See __init__.

        """
        
        columns = []
        explanations = []
        
        # First, handle the categorical covariates.
        features_and_defaults = zip(
            self.additional_subject_categorical_covariates,
            self.additional_covariate_default_states
        )

        for feature, default_value in features_and_defaults:
            try:
                values = set(self.subject_data[feature])
            except KeyError():
                # The default exception message benefits from
                # a little context here
                raise KeyError('Trying to control for covariate %s, but no "%s" '
                               'column in subject_data.' % (feature, feature))
            try:
                other_values = values.copy()
                other_values.remove(default_value)
            except KeyError:
                raise ValueError('Trying to control for covariate %s, but no '
                                 'subject has the default value "%s". To avoid '
                                 'identifiability problems, at least one subject '
                                 'must have the default value.' %
                                 (feature, default_value))
            # With that out of the way...
            for alternative in other_values:
                explanations.append((feature, alternative))
                columns.append(
                    (self.subject_data[feature] == alternative).values.astype('int64')
                )
        # Second, the continuous covariates.
        for feature in self.additional_subject_continuous_covariates:
            try:
                values = self.subject_data[feature].values
            except KeyError():
                # The default exception message benefits from
                # a little context here
                raise KeyError('Trying to control for covariate %s, but no "%s" '
                               'column in subject_data.' % (feature, feature))
            columns.append(values - np.mean(values))
            explanations.append('continuous feature %s' % feature)

        self.additional_covariate_encoding = tuple(explanations)
        if columns:
            self.additional_covariate_matrix = np.vstack(columns).T
        else:
            self.additional_covariate_matrix = None
        self.n_fixed_covariates = 1 + len(columns)

    def copy(self):
        return copy.deepcopy(self)

    def apply_rules(self, rule_list):
        """ Tabulate which rules apply to which subjects.

        Returns an n_rules x n_subjects array of booleans.

        """
        # Don't use ufunc.reduce here, it is slower
        rule_results = [
            reduce(np.logical_and, [self.apply_primitive(p) for p in rule]) for
            rule in rule_list.rules
        ]
        if rule_results:
            return np.vstack(rule_results)
        else:
            return []

    def covariate_matrix(self, rule_list):
        """ Calculate covariate matrix resulting when a rule list is applied to this data.

        A convenience function, returning
        self.apply_rules(rule_list).T after casting it to integer
        dtype (from boolean)-- this is important for cases where the
        dot product of the covariate matrix with its transpose is
        taken, for example-- and appending a column of 1s
        (representing the 'default rule' or a constant term in the
        logistic regression case) as well as self.additional_covariate_matrix
        (if applicable).

        If the rule list is empty, the matrix includes only the constant terms.

        """
        if len(rule_list) < 1:
            X = np.ones((self.n_subjects, 1),dtype=np.int64)

        else:
            # Multiplying by 1 promotes the matrix to integer. The hstack
            # might too, though I have not checked this
            X = np.hstack((1 * self.apply_rules(rule_list).T, 
                           np.ones((self.n_subjects, 1),dtype=np.int64),)
                          )
        if self.additional_covariate_matrix is not None:
            X = np.hstack((X,self.additional_covariate_matrix))
            
        return X

    def apply_primitive(self, primitive):
        if primitive.as_tuple() in self._primitive_result_cache:
            return self._primitive_result_cache[primitive.as_tuple()]
        else:
            return self._apply_primitive(primitive)

    def _apply_primitive(self, primitive):
        values = []
        for subject_x,subject_t in zip(self.X, self.T):
            values.append(primitive.apply(subject_x,subject_t))
        return np.array(values)                          

    def stratify(self, rule_list):
        subjects_handled = np.zeros(self.n_subjects, 
                                    dtype='bool')
        class_memberships = np.zeros((len(rule_list.rules) + 1,
                                      self.n_subjects),
                                     dtype='bool')
        i = -1 # need to make this explicit for the empty-rule-list case
        for i, rule_results in enumerate(self.apply_rules(rule_list)):
            this_rule_subjects = rule_results & (~subjects_handled)
            class_memberships[i,:] = this_rule_subjects
            subjects_handled = subjects_handled | this_rule_subjects
        class_memberships[i+1,:] = ~subjects_handled
        return class_memberships

    def y_by_class(self,rule_list):
        class_memberships = self.stratify(rule_list)
        return zip(np.sum(class_memberships*self.y,1),
                   np.sum(class_memberships,1))

    def __str__(self):
        template = ('<Dataset with %d subjects ' +
                    '(%d observations of %d variables)>')
        n_obs = sum([len(timepoints) for timepoints in self.T])
        return template % (self.n_subjects, n_obs,
                           self.n_variables)

def load_abundance_data(filename, strip_quotes=True):
    """ Read a CSV of abundances, returning a data frame.

    We expect a CSV formatted like:
    
    "", "GCA...GGG", "GCA...TGG"
    "sampleID1", 5, 0
    "sampleID2", 400, 25
    
    (though the data need not be integers.) 

    A pandas dataframe is returned. 
    By default, (double) quotes are stripped from the sample names
    and OTU identifiers.

    """

    # It is important that the sample IDs, thus the 
    # index column of this dataframe, be strings, 
    # so we specify a converter function for column 0.
    df = pd.read_csv(filename,index_col=0,
                     converters={0:lambda x: str(x)})
    if strip_quotes:
        df.rename(index=dequote, columns=dequote)
    return df


def fasta_to_dict(filename):
    """ Read a mapping of IDs to sequences from a fasta file.

    For relabeling DADA2 RSVs. Returns a dict {sequence1: name1, ...}.

    """
    with open(filename) as f:
        whole = f.read()
    pairs = whole.split('\n>')
    table = {}
    # remove leading '>'
    pairs[0] = pairs[0][1:]
    for pair in pairs:
        name, sequence = pair.strip().split('\n')
        table[sequence] = name
    return table


def load_sample_metadata(filename, strip_quotes=True):
    """ Read subject, time, and optionally other data.

    Expects a three-column CSV with no header row.

    First column the sample ID, second column a unique identifier
    (treated as a string) for each subject, third column an observation
    time (treated as a float.)

    In the future, additional time-varying data measured at each
    sampling point will be accepted as additional columns.

    If strip_quotes is True (the default), remove quotation marks
    around the sample IDs if there are any.

    Returns a list of tuples, one for each subject:
    [(subject_identifier, [(timepoint1, sampleID1), ... ]), ...]
    
    To match pandas behavior, if _all_ subject identifiers appear
    to be integers, they are cast to int (from str.) 

    """
    # In the future load a data frame and iterate through the rows.
    table = {}
    with open(filename) as f:
        for line in f:
            sample, subject, time = line.strip().split(',')
            time = float(time)
            if strip_quotes:
                sample = dequote(sample)
            table.setdefault(subject,[]).append((time, sample))
    try:
        table = {int(k): v for k,v in table.items()}
    except ValueError:
        pass
    for subject in table:
        table[subject].sort()
    return sorted(table.items())


def load_subject_data(filename):
    """ Read non-time-dependent data for each subject from csv.

    The first column must be the subject identifier used
    in the sample metadata file. The first row should give 
    field names (field name for the first column is not important.) 

    """
    df = pd.read_csv(filename,index_col=0)
    return df


def load_16S_result(abundance_data_filename, 
                      sample_metadata_filename, subject_data_filename,
                      sequence_id_filename=None,
                      **kwargs):
    abundances = load_abundance_data(abundance_data_filename)
    if sequence_id_filename is not None:
        sequence_names = fasta_to_dict(sequence_id_filename)
        abundances = abundances.rename(index={}, columns=sequence_names)
    sample_metadata = load_sample_metadata(sample_metadata_filename)
    subject_data = load_subject_data(subject_data_filename)
    return combine_data(abundances, sample_metadata,
                        subject_data,
                        **kwargs)


def combine_data(abundance_data,
                 sample_metadata,
                 subject_data,
                 experiment_start=None,
                 experiment_end=None,
                 outcome_variable=None,
                 outcome_positive_value=None,
                 additional_subject_categorical_covariates=[],
                 additional_covariate_default_states=[],
                 additional_subject_continuous_covariates=[]):
    """ Assemble data loaded from files into appropriate Dataset.

    abundance_data (a pandas.Dataframe), sample_metadata (a list),
    and subject_data (a pandas.Dataframe) should be formatted
    as the output of load_abundance_data, load_sample_metadata,
    and load_subject_data.

    If experiment_start is given, it will be passed to the Dataset
    constructor; otherwise the minimum observation time is used.
    Similarly for experiment_end.

    If outcome_variable and outcome_positive_value are given,
    subjects where subject_data[outcome_variable] == outcome_positive_value
    have y=1, and subjects where subject_data[outcome_variable] is NaN
    are excluded. Otherwise, y=0 for all subjects.

    If additional_subject_categorical_covariates are given, subjects
    where subject_data[variable] is NaN for any variable in the list
    are excluded. The same is done for additional_subject_continuous_covariates.

    The variable_prior is set to 1.0 for each variable.

    """

    # Assemble X and T and the list of subject IDs.
    X = []
    T = []
    subject_IDs = []
    skipped_samples = []
    for subject, samples in sample_metadata:
        subject_IDs.append(subject)
        this_subject_timepoints = []
        this_subject_observations = []
        for timepoint, sample_id in samples:
            try:
                values = abundance_data.loc[sample_id].values
            except KeyError:
                skipped_samples.append(sample_id)
                continue
            this_subject_timepoints.append(timepoint)
            this_subject_observations.append(
                abundance_data.loc[sample_id].values
            )
        T.append(np.array(this_subject_timepoints))
        # For whatever reason, convention is that
        # X entries have OTUs along first axis and
        # timepoints along second axis
        if this_subject_observations:
            X.append(np.vstack(this_subject_observations).T)
        else:
            X.append(np.array([]))
        
    # Extract variable names, count them, set up prior.
    variable_names = [s for s in abundance_data.columns]
    variable_prior = np.ones(len(variable_names))
    
    # Establish experiment start and end times.
    max_observation_times = []
    min_observation_times = []
    for timepoints in T:
        # Pathological subjects may have no observations
        if len(timepoints) == 0:
            continue
        max_observation_times.append(max(timepoints))
        min_observation_times.append(min(timepoints))
    # If we haven't found a valid observation for any subject
    # we cannot proceed.
    if not min_observation_times:
        raise ValueError("Could not locate data for any specified sample- is your abundance data matrix properly formatted, with a row for each sample specified in the sample metadata table?")
    if experiment_start is None:
        experiment_start = min(min_observation_times)
    if experiment_end is None:
        experiment_end = max(max_observation_times)

    if outcome_variable is not None and outcome_positive_value is not None:
        outcome_column = subject_data.loc[:,outcome_variable]
        sorted_outcomes = []
        for subject in subject_IDs:
            sorted_outcomes.append(
                outcome_column.loc[subject] == outcome_positive_value
            )
        y = np.array(sorted_outcomes)
        if len(y) != len(subject_IDs):
            raise ValueError('Wrong number of outcome values provided. Please check that the subject metadata file contains one line per subject.')
            
    else:
        y = np.zeros(len(subject_IDs))    
        
    result = MITREDataset(X,T,y,variable_names,variable_prior,
                     experiment_start, experiment_end,
                     subject_IDs, subject_data,
                     additional_subject_categorical_covariates = additional_subject_categorical_covariates,
                     additional_subject_continuous_covariates = additional_subject_continuous_covariates,
                     additional_covariate_default_states = additional_covariate_default_states
                     )

    if outcome_variable is not None:
        result = discard_where_data_missing(result,
                                                   outcome_variable) 
    for variable in additional_subject_categorical_covariates:
        result = discard_where_data_missing(result,
                                                      variable) 
    for variable in additional_subject_continuous_covariates:
        result = discard_where_data_missing(result,
                                                      variable) 
    # Note that the NaNs may have led to spurious or erroneous columns
    # in the covariate matrix which will go away if we regenerate it.
    if (additional_subject_categorical_covariates or
        additional_subject_continuous_covariates):
        result.generate_additional_covariate_matrix()
        
    return result


def load_metaphlan_abundances(abundance_file):
    """ Reformat a Metaphlan output table.

    Assumes abundance_file is the name of a tab-delimited
    file, one row per clade, one column per sample.

    Changes percentages to fractions and transposes the table,
    returning a DataFrame.

    """
    raw = pd.read_table(abundance_file,index_col=0)
    return 0.01*raw.T


def load_metaphlan_result(abundance_data_filename,
                          sample_metadata_filename,
                          subject_data_filename,
                          do_weights=False,
                          weight_scale = 1.0,
                          **kwargs):
    abundances = load_metaphlan_abundances(abundance_data_filename)
    assert 'k__Bacteria' in abundances.columns
    sample_metadata = load_sample_metadata(sample_metadata_filename)
    subject_data = load_subject_data(subject_data_filename)
    data = combine_data(abundances, sample_metadata,
                        subject_data,
                        **kwargs)
    # Create the variable tree and tweak the prior here
    names_to_nodes = {}
    for name in abundances.columns:
        names_to_nodes[name] = Tree(name=name, dist=1.0)
    for k,v in names_to_nodes.items():
        taxonomy = k.split('|')
        if len(taxonomy) == 1:
            continue
        parent = '|'.join(taxonomy[:-1])
        parent_node = names_to_nodes[parent]
        parent_node.add_child(v)
    root = names_to_nodes['k__Bacteria']
    if do_weights:
        for i,v in enumerate(data.variable_names):
            if v in names_to_nodes:
                data.variable_weights[i] = (
                    weight_scale *
                    (1 + len(names_to_nodes[v].get_descendants()))
                )
        
    data.variable_tree = root
    
        
    return data


def describe_dataset(dataset, comment=None):
    """ Log size of a dataset object. 

    A utility function, used often in the preprocessing step.

    """
    if comment is not None:
        print(comment)
    print('%d variables, %d subjects, %d total samples' % 
          (dataset.n_variables, dataset.n_subjects,
           sum(map(len, dataset.T))))


def take_relative_abundance(data):
    """ Transform abundance measurements to relative abundance. """

    new_data = data.copy()
    n_subjects = len(new_data.X)
    for i in range(n_subjects):
        abundances = new_data.X[i]
        # Data may be integer (counts): cast carefully
        # to float
        total_abundances = np.sum(abundances, axis=0).astype(np.float)
        relative_abundances = abundances/total_abundances
        new_data.X[i] = relative_abundances
    return new_data


def do_internal_normalization(data,
                              target_variable_names,
                              reject_threshold=1e-6):
    """ Normalize abundance measurements by sum of some variables. """
    try:
        target_indices = [data.variable_names.index(n) for n in
                            target_variable_names]
    except ValueError:
        raise ValueError(
            'Variable name %s specified for use in internal normalization,'
            ' but not found in data. Double-check it is a valid name, and'
            ' has not been accidentally removed by filtering settings.' %
            n
        )
    new_data = data.copy()
    n_subjects = len(new_data.X)
    for i in range(n_subjects):
        abundances = new_data.X[i]
        target_abundances = abundances[target_indices, :]
        # Data may be integer (counts): cast carefully
        # to float
        norm_factors = np.sum(target_abundances, axis=0).astype(np.float)
        if not np.all(norm_factors > reject_threshold):
            bad_indices = np.where(norm_factors <= reject_threshold) 
            bad_timepoints = data.T[i][bad_indices]
            subject_id = data.subject_IDs[i]
            message = (
                'Error normalizing data for subject %s: '
                'sum of variables used for normalization is less than '
                'the minimum %.3g at timepoints %s'
                % (subject_id, reject_threshold,
                   ','.join(['%.3g' % t for t in bad_timepoints])
                )
            )
            raise ValueError(message)
        normalized_abundances = abundances/norm_factors
        new_data.X[i] = normalized_abundances
    return new_data


def get_normalization_variables(config, data):
    """ Figure out which variables to normalize by, if relevant.
    
    If preprocessing/normalization_variables_file is set, loads that
    file and reads a variable name from each line.

    Returns a list of variable names (as strings).

    """
    if (config.has_option('preprocessing', 'normalization_variables_file')
        and
        config.has_option('preprocessing', 'normalize_by_taxon')):
        raise ValueError('Mutually exclusive normalization options given.')
    
    if config.has_option('preprocessing', 'normalization_variables_file'):
        filename = config.get(
            'preprocessing',
            'normalization_variables_file'
        )
        with open(filename) as f:
            variables = [s.strip() for s in f.readlines()]
        # Tolerate extra newlines, etc
        result = [v for v in variables if v]
        print(
            'Read %d variables for normalization from %s' %
            (len(result), filename)
        )
        return result
    elif config.has_option('preprocessing','normalize_by_taxon'):
        if not config.has_option('data','placement_table'):
            raise ValueError(
                'A taxonomic placement table must be '
                'specified to allow normalization by a particular taxon.'
            )
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        table = taxonomy_annotation.load_table(
            placement_table_filename,
            sequence_fasta_filename
        )
        target_taxon = config.get('preprocessing','normalize_by_taxon')
        target_variables = []
        for v in data.variable_names:
            classifications = table.loc[v,:]
            if target_taxon in classifications.values:
                target_variables.append(v)
        if not target_variables:
            raise ValueError(
                'Found no variables in designated normalization '
                'taxon "%s".' % target_taxon
            )
        prefix = config.get('description', 'tag')
        fname = prefix + '_variables_used_for_normalization.csv'
        subtable = table.loc[target_variables,:]
        subtable.to_csv(fname)
        print(
            '%d variables in designated normalization taxon '
            '%s found (see %s)' % (len(target_variables),
                                   target_taxon,
                                   fname)
        )
        return target_variables
    else:
        raise ValueError(
            'Must set normalization_variables_file or '
            'normalization taxon in section '
            '"preprocessing" to do internal normalization.')


def preprocess(config):
    """ Load data, apply filters, create Dataset object.
    
    This is broken into two parts: a first step which 
    loads the data, applies initial filters, and converts to 
    relative abundance; then a second step which performs 
    phylogenetic aggregation and final filtering.

    """
    data = preprocess_step1(config)
    data = preprocess_step2(config, data)
    return data


def preprocess_step1(config):
    """ Load data, apply initial filters and convert to RA, create Dataset object.

    If specified, generate simulated data based on the data that would 
    otherwise be loaded, and return (after processing) that dataset instead.

    """
    # 0. If necessary, update the configuration to contain the appropriate 
    # settings for one of the example data sets.

    if config.has_option('data', 'load_example'):
        load_example(config, config.get('data','load_example'))

    # 1. Input files. 
    counts_file = config.get('data','abundance_data')
    metadata_file = config.get('data','sample_metadata')
    subject_file = config.get('data', 'subject_data') 
    if config.has_option('data','sequence_key'):
        sequence_file = config.get('data','sequence_key')
    else:
        sequence_file = None

    # 2. Outcome
    outcome_variable = config.get('data', 'outcome_variable') 
    outcome_positive_value = config.get('data', 'outcome_positive_value') 
    # We don't know whether to expect the positive outcome value to be
    # a string, boolean, or integer, but the data type needs to match
    # the type in the dataframe of per-subject data, at least enough to 
    # allow meaningful equality testing. Somewhat clumsily, we 
    # just try to cast the string we read from the file to an int;
    # if the true value is Boolean, specify either 1 or 0 in the 
    # configuration file (not, e.g., 'true' or 'false').

    try:
        outcome_positive_value = int(outcome_positive_value)
    except ValueError:
        if outcome_positive_value.lower() in ('true','false'):
            message = ('Boolean outcome values should specified as 1 or 0 '
                       '(not the strings "true", "false", or derivatives of these) '
                       '- the currently specified value will be interpreted as a '
                       'generic categorical value which will probably lead to '
                       'undesirable behavior!')
            warnings.warn(message)
        pass

    # 2a. Additional covariates. Assume that these are provided as
    # comma-separated lists. 

    # First, the categorical covariates. For categorical data, try to
    # convert strings to ints if possible (this should roughly match
    # the behavior of the import of the subject data file.)
    if config.has_option('data','additional_subject_covariates'):
        additional_subject_covariates = config.get('data','additional_subject_covariates')
        additional_subject_covariates = additional_subject_covariates.split(',')
        raw_default_states = config.get('data','additional_covariate_default_states')
        raw_default_states = raw_default_states.split(',')
        additional_covariate_default_states = []
        for state in raw_default_states:
            try: 
                state = int(state)
            except ValueError:
                if state.lower() in ('true','false'):
                    message = (
                        'Boolean default states should specified as 1 or 0 '
                        '(not the strings "true", "false", or derivatives of these) '
                        '- the currently specified value will be interpreted as a '
                        'generic categorical value which will probably lead to '
                        'undesirable behavior!'
                    )
                    warnings.warn(message)
                pass
            additional_covariate_default_states.append(state)
    else:
        additional_covariate_default_states = []
        additional_subject_covariates = []

    # Second, the continuous covariates.  
    if config.has_option('data','additional_subject_continuous_covariates'):
        additional_subject_continuous_covariates = config.get(
            'data',
            'additional_subject_continuous_covariates'
        )
        additional_subject_continuous_covariates = additional_subject_continuous_covariates.split(',')
    else:
        additional_subject_continuous_covariates = []

    # Loading the data. This depends on what type of
    # data we have.
    data_type = '16s'
    if config.has_option('data', 'data_type'):
        data_type = config.get('data','data_type').lower()
    assert data_type in ('16s','metaphlan')
    if data_type == 'metaphlan':
        do_weights = False
        weight_scale = 1.0
        if config.has_option('data', 'metaphlan_do_weights'):
            do_weights = config.getboolean('data','metaphlan_do_weights')
        if config.has_option('data', 'metaphlan_weight_scale'):
            weight_scale = config.getfloat('data','metaphlan_weight_scale')
        data = load_metaphlan_result(
            counts_file,
            metadata_file,
            subject_file,
            do_weights = do_weights,
            weight_scale = weight_scale,
            outcome_variable=outcome_variable,
            outcome_positive_value=outcome_positive_value, 
            additional_subject_categorical_covariates = additional_subject_covariates,
            additional_covariate_default_states = additional_covariate_default_states,
            additional_subject_continuous_covariates = additional_subject_continuous_covariates,
        )

    else: # default to assuming 16s
        data = load_16S_result(
            counts_file,
            metadata_file,
            subject_file,
            sequence_id_filename=sequence_file,
            outcome_variable=outcome_variable,
            outcome_positive_value=outcome_positive_value, 
            additional_subject_categorical_covariates = additional_subject_covariates,
            additional_covariate_default_states = additional_covariate_default_states,
            additional_subject_continuous_covariates = additional_subject_continuous_covariates,
        )
    
    if len(np.unique(data.y)) == 1:
        message = ('All subjects have the same outcome. No model can be trained '
                   'based on this data. Double-check that the outcome variable '
                   ' is correctly encoded?')
        warnings.warn(message)
    describe_dataset(data, 'Data imported (before any filtering:)')

    # 3. Filtering

    # 3a. Overall abundance filter
    if config.has_option('preprocessing','min_overall_abundance'):
        # Drop sequences/OTUs with fewer reads (summing across all
        # samples) than the threshold
        minimum_reads_per_sequence = config.getfloat(
            'preprocessing','min_overall_abundance'
        )
        data, _ = discard_low_overall_abundance(
            data,
            minimum_reads_per_sequence
        )
        describe_dataset(
            data,
            'After filtering RSVs/OTUs with too few counts:'
        )

    # 3b. Sample depth filter
    if config.has_option('preprocessing','min_sample_reads'):
        # Drop all samples where the total number of reads was below a
        # threshold
        minimum_reads_per_sample = config.getfloat(
            'preprocessing',
            'min_sample_reads'
        )
        data = discard_low_depth_samples(
            data,
            minimum_reads_per_sample
        )
        describe_dataset(
            data,
            'After filtering samples with too few counts:'
        )

    # 3c. Trimming the experimental window
    if config.has_option('preprocessing','trim_start'):
        experiment_start = config.getfloat('preprocessing','trim_start')
        experiment_end = config.getfloat('preprocessing','trim_stop')
        data = trim(data, experiment_start, experiment_end)
        describe_dataset(
            data,
            'After trimming dataset to specified experimental time window:'
        )

    # 3d. Drop subjects with inadequately dense temporal sampling
    if config.has_option('preprocessing','density_filter_n_samples'):
        subject_min_observations_per_long_window = (
            config.getfloat('preprocessing',
                            'density_filter_n_samples')
        )
        n_intervals = config.getint(
            'preprocessing',
            'density_filter_n_intervals')
        n_consecutive = config.getint(
            'preprocessing',
            'density_filter_n_consecutive'
        )
        data = filter_on_sample_density(
            data,
            subject_min_observations_per_long_window,
            n_intervals,
            n_consecutive=n_consecutive
        )
        describe_dataset(
            data,
            ('After filtering subjects with ' + 
            'inadequately dense temporal sampling:')
        )

    # Optionally subsample the data, keeping a set number of subjects
    # chosen at random.
    if config.has_option('preprocessing','subsample_subjects'):
        n_subjects_to_keep = config.getint('preprocessing','subsample_subjects')
        indices_to_keep = np.random.choice(data.n_subjects, n_subjects_to_keep, replace=False)
        data = select_subjects(data, indices_to_keep)
        print('Subsampling, kept indices: %s' % str(indices_to_keep))

    # 3e. Relative abundance transformation, or other normalization.
    if (config.has_option('preprocessing', 'take_relative_abundance') and
        config.has_option('preprocessing', 'do_internal_normalization')):
        if (config.getboolean('preprocessing', 'take_relative_abundance') and
            config.getboolean('preprocessing', 'do_internal_normalization')):
            raise ValueError(
                'Cannot both take relative abundance and do '
                'internal normalization.'
                )
    
    if config.has_option('preprocessing', 'take_relative_abundance'):
        if config.getboolean('preprocessing','take_relative_abundance'):
            data = take_relative_abundance(data) 
            print('Transformed to relative abundance.')

    if config.has_option('preprocessing', 'do_internal_normalization'):
        if config.getboolean('preprocessing','do_internal_normalization'):
            normalization_variables = get_normalization_variables(config, data)
            if config.has_option('preprocessing',
                                 'internal_normalization_min_factor'):
                threshold = config.getfloat(
                    'preprocessing',
                    'internal_normalization_min_factor'
                )
            else:
                threshold = 1.0
            data = do_internal_normalization(
                data, normalization_variables, reject_threshold=threshold
            ) 
            print('Internal normalization complete.')

    return data


# Example Newick strings for debugging.
standard = "((A:.01[e], B:.01)D:.01[g], C:.01[h]);"
edged = "((A:.01[e]{0}, B:.01{1})D:.01{3}[g], C:.01{4}[h]){5};"
typical = "((A:.01{0}, B:.01{1})1.0:.01{3}, C:.01{4}){5};"

edge_number_pattern = re.compile(
    r'([^{}():,]+):([^{}():,]+)\{([^{}():,]+)\}'
)
terminal_edge_number_pattern = re.compile(r'\{([^{}():,]+)\};$')

def load_jplace(filename):
    with open(filename) as f:
        jplace = json.load(f)
    return jplace


def reformat_tree(jplace):
    """ Convert edge-numbered Newick to (more) conventional Newick.

    We expect an input tree of the form:

    ((A:.01{0}, B:.01{1})1.0:.01{3}, C:.01{4}){5};

    (in jplace['tree']- jplace should be the output of load_jplace)
    and wish to obtain a tree with unique internal node identifiers,
    but without edge numbering. We also don't care about internal node
    support values.
    
    We achieve this by reidentifying each node (internal and leaf) 
    with its edge number (that is, the edge number of the edge 
    connecting it to its parent:)

    (("0":.01, "1":.01)"3":.01, "4":.01)"5"; 

    The resulting Newick string is read into an ete3.Tree and that is 
    returned as the first value.

    We also keep track of the original names (for leaves) or 
    support values (for internal nodes) and return two dictionaries:
    one mapping original leaf names to new ids, and one mapping
    new ids to collections of the original names of all descendant
    leaves. Note the edge numbers appear in these dictionaries as strings,
    whether they are keys or values, eg {'0': A, ...}. 

    """
    tree_string = jplace['tree']
    new_id_to_old_value = {}
    for match in edge_number_pattern.finditer(tree_string):
        value, distance, edge_number = match.groups()
        new_id_to_old_value[edge_number] = value

    relabel = lambda match: match.group(3) + ':' + match.group(2)

    new_tree, _ = edge_number_pattern.subn(relabel,tree_string)
    new_tree = terminal_edge_number_pattern.sub(
        lambda match: '%s;' % match.group(1),new_tree
    )

    new_tree = Tree(new_tree, format=1)
    
    leaf_to_new_id = {}
    new_id_to_old_leaf_names = {} 

    for node in new_tree.traverse(strategy='levelorder'):
        name = node.name
        if node.is_leaf():
            leaf_to_new_id[new_id_to_old_value[name]] = name
        old_names_of_leaves = []
        for leaf in node.get_leaves():
            old_names_of_leaves.append(
                new_id_to_old_value[leaf.name]
            )
        new_id_to_old_leaf_names[name] = old_names_of_leaves
        
    return new_tree, leaf_to_new_id, new_id_to_old_leaf_names

def organize_placements(jplace_as_dict):
    """ Extract simple list of placement candidates from jplace import.

    Input should be the result of load_jplace.

    Output: 

    {'placed_sequence_name_0': [(likelihood_weight_ratio_01, edge_number_01, distal01, pendant01),
                                (likelihood_weight_ratio_02, edge_number_02, distal02, pendant02), 
                                ...],
     ...}

    Edge numbers, which in the current somewhat recondite scheme become
    node names, are converted to strings.

    """
    placements = {}
    fields = jplace_as_dict['fields']
    for entry in jplace_as_dict['placements']:
        keys = [l[0] for l in entry['nm']]
        this_sequence_placements = []
        for record in entry['p']:
            attributes = dict(zip(fields,record))
            this_sequence_placements.append(
                (attributes['like_weight_ratio'],
                 str(attributes['edge_num']),
                 attributes['distal_length'],
                 attributes['pendant_length'])
            )
        this_sequence_placements.sort()
        for key in keys:
            placements[key] = this_sequence_placements[:]
    return placements


def extract_weights_simplified(tree, placements, target_sequences, prune_before_weighting=True):
    """ Identify OTU ancestors, subtree weights.

    Like extract_weights, but assigns each OTU to a definite best
    ancestor, rather than accounting for all possible placements. 

    See aggregate_by_pplacer_simplified for a description of how
    this is done.

    Arguments:
    tree, placements, target_sequences: See exact_weights.
    
    prune_before_weighting: if True (the default), prune the tree to
    include only OTUs and the higher nodes necessary to preserve
    topological relationships among them _before_ calculating subtree
    weight values. Otherwise, do this afterwards.
    
    Returns:
    sequence_to_ancestors - dict mapping each target sequence to 
    a list of nodes ancestral to its best immediate ancestor, including 
    the best immediate ancestor itself.

    node_to_weight - dict mapping each node in the tree to the
    estimated length of the subtree descending from the edge
    immediately above that node, and each target sequence to the
    estimated length of the subtree including only the edge connecting
    it to the reference tree. N.B.: as in extract_weights, the subtree
    weights assigned to ordinary edges include contributions from
    sequences which may be placed on the subtrees descending from
    those edges. Where above, these contributions were weighted by the
    estimated probability that the sequences are placed there, here we
    instead treat placements as definite. However, subtree weights for
    target sequences do represent a weighted average over all
    placements.

    tree - the tree used to obtain weights, with nodes for OTUs added
    as children of their immediate best ancestor (yes, this is
    conceptually odd as placement is made to edges, not nodes of the
    reference tree, but this tree object describes the hierarchical
    relationship among variables in the model, rather than
    recapitulating a best-estimate phylogeny exactly.) The tree is
    pruned to include only OTUs and the higher nodes necessary to
    preserve the topology.

    node_to_sequences - dict mapping the name of each node to a list 
    of OTUs that descend from it.

    """
    sequence_to_best_ancestors = {}
    sequence_to_all_ancestors = {}
    sequence_typical_pendant_length = {}
    node_to_weight = {}
    node_to_sequences = {}
    # Don't destructively modify the input:
    tree = tree.copy()

    target_sequence_set = set(target_sequences)
    # Rely on the assumption that all nodes have unique names.
    name_to_node = {n.name: n for n in tree.get_descendants()}
    name_to_node[tree.name] = tree

    for sequence in target_sequences:
        # placements[sequence] is a list of tuples, each of the form
        # (likelihood_weight_ratio, edge, distal, pendant) 
        ancestry_probabilities_this_sequence = {}
        weighted_pendant_length = 0
        best_ancestor = False

        for weight, edge, distal, pendant in placements[sequence]:
            weighted_pendant_length += weight * pendant
            ancestor = name_to_node[edge]
            for node in ([ancestor] + ancestor.get_ancestors()):
                ancestry_probabilities_this_sequence.setdefault(node.name, []).append(weight)
            if weight > 0.5:
                best_ancestor = edge
        sequence_typical_pendant_length[sequence] = weighted_pendant_length

        if not best_ancestor:
            sorted_ancestors = sorted(
                [(sum(weights), k) for k, weights in 
                 ancestry_probabilities_this_sequence.items() if
                 sum(weights) > 0.5]
            )
            best_ancestor = sorted_ancestors[0][1]

        sequence_to_best_ancestors[sequence] = best_ancestor
        all_ancestors = (
            [best_ancestor] +
            [n.name for n in name_to_node[best_ancestor].get_ancestors()]
        )
        sequence_to_all_ancestors[sequence] = all_ancestors
        for ancestor in all_ancestors:
            node_to_sequences.setdefault(ancestor,[]).append(sequence)
        
    # Add OTUs as children of best ancestors.
    print('Attaching sequences/OTUs to tree...')
    for otu, best_ancestor in sequence_to_best_ancestors.items():
        dist = sequence_typical_pendant_length[otu]
        best_ancestor_as_node = name_to_node[best_ancestor]
        best_ancestor_as_node.add_child(name=otu,dist=dist)

    if prune_before_weighting: 
        print('Pruning (this may take a moment)...')
        tree.prune(target_sequences, preserve_branch_length=True)

    # walk from leaves back up
    # To avoid confusion with the placement weights, refer to these as
    # lengths, even though they are only sort of lengths.
    print('Calculating weights...')
    for node in reversed(list(tree.traverse(strategy='levelorder'))):
        length = node.dist 
        length += sum(node_to_weight[child.name] for child in node.children)
        node_to_weight[node.name] = length

    if not prune_before_weighting: 
        tree.prune(target_sequences, preserve_branch_length=True)

    # Discard data pertaining to nodes that have been filtered out of the tree
    remaining_nodes = {n.name for n in tree.get_descendants()}
    remaining_nodes.add(tree.name)
    sequence_to_all_ancestors = {s: [ancestor for ancestor in v if ancestor in remaining_nodes] for
                                 s,v in sequence_to_all_ancestors.items()}
    node_to_weight = {k:v for k,v in node_to_weight.items() if k in remaining_nodes}
    node_to_sequences = {k:v for k,v in node_to_sequences.items() if k in remaining_nodes}

    return sequence_to_all_ancestors, node_to_weight, tree, node_to_sequences


def aggregate_by_pplacer_simplified(jplace_filename, input_dataset):
    """ Aggregation on phylogenetic tree using pplacer results (simple)  

    This proceeds much as aggregate_by_pplacer. However, instead of
    tracking all the separate possible placement locations for each
    OTU, each OTU is given a single placement to the lowest node in
    the tree with probability of being an ancestor of the OTU (as
    estimated by summing the LWR of placements to the node's children)
    is greater than 0.5. The OTUs are then added as child nodes of the
    nodes to which they are placed (this is a little odd conceptually,
    as placement is notionally to the edge _above_ the node, but
    accurately reflects the hierarchical relationship we use to
    aggregate data, etc.) The edge length of the edge connecting the
    OTU to its ancestor node is the weighted average pendant length of
    that node's placements (even if the chosen ancestor node should
    correspond to a particular placement with pendand length not
    exactly equal to that average.) Data for the OTU is aggregated to
    the best ancestor and its ancestors in turn as though each had a
    combined LWR=1.0.
    
    Some subtleties: all variable weights are the same as they 
    would be under the more complicated placement process. 

    Note the abundance of nodes lying below a node's best ancestor may
    be a little odd: consider the tree

      /-C
    A-|
      \-B 

    and suppose OTU1 attaches with equal probability to B or C, but
    OTU2 certainly attaches to C, as does OTU3 to B. A will be the
    best ancestor of OTU1, and the data for C will reflect only the
    abundance of OTU2 and B only the abundance of OTU3, even though
    OTU1 certainly must attach to one or the other. We contend this is an
    acceptable approximation: the 'abundance of subtree B [or C]' does
    not exist independent of some defined procedure for mapping reads
    to tree nodes, and this is how we do it. Close inspection of
    results will in any case reveal that a rule attaching to C refers
    to a group of bacteria including OTU2 but not OTU1.

    Return and other behavior the same as for aggregate_by_pplacer.

    """
    print('Loading and reprocessing phylogenetic placements...')
    jplace = load_jplace(jplace_filename)
    tree, _, _  = reformat_tree(jplace)
    placements = organize_placements(jplace)
    target_sequences = list(
        set(placements).intersection(input_dataset.variable_names)
    )

    print('Identifying best placements and calculating subtree weights...')
    result = extract_weights_simplified(
        tree, placements, target_sequences, prune_before_weighting=True
    )
    sequence_to_ancestors, var_to_weight, tree_of_variables, node_to_sequence = result
   
    # Comments on adding the OTUs to the tree.
    # 
    # We need a cleaner way to map nodes back to their descendant
    # input sequences. We also generally want to only assign rules to
    # specific OTUs if the OTU level gives better information than
    # their apparent position on the tree-- if we incorporate the OTUs
    # into the tree, we can achieve this through the tree-based
    # redundancy filtering in cases where an OTU's best ancestor is
    # not the best ancestor of any other OTU suppressed by redundancy
    # filtering. (In cases where a node is the best ancestor of
    # multiple OTUs, we definitely want to preserve the node and the
    # OTUs as separate variables.) So there are good reasons to add
    # each OTU as a child of its immediate best ancestor- it is
    # conceptually a little weird, as in fact the OTU may notionally
    # attach to the reference tree on the edge _above_ the 'ancestor',
    # but we are building a tree that represents a hierarchy of groups
    # of variables, not strictly a phylogeny, for all that the groups
    # and their relationships are almost all phylogenetically defined.
    # (We could alternatively add a new node on the edge above the
    # best ancestor, but then we get into questions of what the
    # appropriate edge length would be in the case where the best
    # ancestor we are confident in is not actually one of the edges to
    # which a possible placement was identified by pplacer.)

    # assume names are unique
    variable_name_to_node = {n.name: n for n in tree_of_variables.get_descendants()}
    variable_name_to_node[tree_of_variables.name] = tree_of_variables

    # Now add variables to the data matrices. We preserve all the old
    # variables in the input dataset, and add the internal nodes of
    # tree_of_variables. Note that here, the data for each node is
    # simply the sum of the data for the OTUs of which it is an
    # ancestor- no need to weight.
    tree_variable_names = [n.name for n in tree_of_variables.get_descendants()]
    # The above doesn't include the root. We don't actually want the
    # root as a variable, but I have assumed it is in other code (and
    # rules applying to it will be filtered out elsewhere.)
    tree_variable_names.append(tree_of_variables.name)
    old_variable_names = set(input_dataset.variable_names)
    added_variables = [v for v in tree_variable_names if
                       v not in old_variable_names]
    old_n_variables = input_dataset.n_variables
    # Note new_n_variables is NOT the number of new variables, and
    # new_variable_names is the new list of names of all the
    # variables, similarly.
    new_n_variables = old_n_variables + len(added_variables) 
    new_variable_names = input_dataset.variable_names + added_variables
    new_variable_indices = dict(
        zip(new_variable_names,
            np.arange(new_n_variables)
        )
    )
    new_X = []
    print('Aggregating data...')
    for array in input_dataset.X:
        _, nt = array.shape
        new_array = np.zeros((new_n_variables, nt))
        new_array[:old_n_variables,:] = array
        ###
        # Here the aggregation code differs from the original function
        for node_name in added_variables:
            descendant_otus = node_to_sequence[node_name]
            node_index = new_variable_indices[node_name]
            for otu in descendant_otus:
                otu_index = new_variable_indices[otu]
                new_array[node_index,:] += new_array[otu_index]
        # 
        ### 
        new_X.append(new_array)   
    print('Finalizing aggregated data...')
    new_dataset = input_dataset.copy()
    new_dataset.n_variables = new_n_variables
    new_dataset.X = new_X
    new_dataset.variable_names = new_variable_names
    new_dataset.variable_weights = np.array(
        [var_to_weight.get(v,1.0) for v in new_variable_names]
    )
    new_dataset.variable_tree = tree_of_variables
    return new_dataset, var_to_weight, node_to_sequence


def annotate_dataset_pplacer(dataset,         
                             jplace_filename,
                             taxa_table_filename,
                             seq_info_filename):
    """ Add taxonomic information to dataset (from pplacer info).

    Here we use the pplacer reference package to label inner nodes of
    the reference tree, then we label leaves (OTUs, etc) according to
    their parent nodes.

    Note this should be applied at a stage in the filtering
    process when all leaves of the trees correspond to OTUs/RSVs 
    in the original observed data, and vice versa- otherwise
    higher groups will be misidentified as OTUs.

    Annotations are stored in dataset.variable_annotations, a
    dictionary keyed by variable name. It is not updated as the
    dataset is transformed, dropping variables.  (As always, it's the
    user's responsibility to ensure names are unique.)

    dataset: rules.Dataset object
    jplace_filename: pplacer output .jplace file 
    taxa_table_filename: pplacer refpkg taxa table
    seq_info_filename: pplacer refpkg seqinfo file

    Returns: none.

    """
    node_labels = describe_tree_nodes_with_taxonomy(
        jplace_filename,
        taxa_table_filename,
        seq_info_filename
    )
    # We're going to traverse this tree with no particular
    # regard for inefficiency.
    annotations = {}
    nodes = list(dataset.variable_tree.iter_descendants())
    nodes.append(dataset.variable_tree)
    for node in nodes:
        if node.is_leaf():
            parent = node.up.name
            parent_clade = node_labels[parent]
            otu_string = 'OTU mapped to %s' % parent_clade
            annotations[node.name] = otu_string
        else:
            n_leaves = len(node)
            clade = node_labels[node.name]
            annotations[node.name] = (
                '%s' % 
                (clade,)
            )
    dataset.variable_annotations = annotations


def describe_tree_nodes_with_taxonomy(
        jplace_filename,
        taxa_table_filename,
        seq_info_filename,
        to_label=None
        ):

    jplace = load_jplace(jplace_filename)
    tree, leaf_to_new_id, new_id_to_old_leaf_names = (
        reformat_tree(jplace)
    ) 

    _, bare_reference_weights, _ = extract_weights(
        tree, placements={}, target_sequences=[],
        prune=False
    ) 

    taxa_table = pd.read_csv(taxa_table_filename,index_col=0)
    tname = lambda j: str(taxa_table.loc[j]['tax_name'])
    seq_info = pd.read_csv(seq_info_filename,index_col=0)
    reference_species = seq_info.join(taxa_table, on='tax_id', how='inner')

    levels = ['phylum','class','order','family','genus','species']
    results = {}
    
    if to_label is not None:
        to_label = set(to_label)

    for node_id, leaves in new_id_to_old_leaf_names.items():
        if to_label is not None and node_id not in to_label:
            continue

        subtable = reference_species.loc[leaves,:]
        taxa = []
        for level in levels:
            values = list(set(subtable[level].values))
            taxa.append(list(map(tname,values)))
            if len(values) > 1:
                break
        if len(taxa[-1]) == 1:
            descriptor = taxa[-1][0] # ' '.join([l[0] for l in taxa[-2:]])
        elif len(taxa) == 1:
            descriptor = 'a clade within phylum ' + ' or '.join(taxa[0])
        else:
            l = len(taxa) - 2
            descriptor = ('a clade within %s %s,'
                          'including representatives of %s %s' %
                          (levels[l], taxa[l][0], 
                           levels[l+1], ', '.join(taxa[l+1])))
        results[node_id] = descriptor
    return results

def extract_weights(tree, placements, target_sequences, prune=True):
    """ Identify OTU ancestors, subtree weights.

    Arguments:
    tree - (first) return value of reformat_tree
    placements - return value of organize_placements
    target_sequences - list of sequences whose placement on the 
    tree is of interest (should match placed sequence identifiers
    in the original jplace file) 

    If prune is True, the reference tree is pruned, retaining only
    those nodes/edges to which placements (of target sequences!) are
    made and those needed to maintain topology on the subtree induced
    by the retained nodes (preserving relative distances) before any
    further calculation is done. 

    Returns:
    sequence_to_ancestors - dict mapping each target sequence to 
    a dict mapping node names in the tree to a number.
    This gives (where nonzero) the probability (estimated from the 
    likelihood weight ratio) that each node so indicated (thus, each 
    edge above each such node) is an ancestor of the sequence.

    node_to_weight - dict mapping each node in the tree to the
    estimated length of the subtree descending from the edge
    immediately above that node, and each target sequence to the
    estimated length of the subtree including only the edge connecting
    it to the reference tree. Subtree length estimation proceeds as
    follows:

    tree - the tree used to obtain weights. If prune is True, this is
    a pruned copy of the input tree; otherwise, it is the input tree
    itself.

    - For an ordinary edge of the reference tree, the length includes 
    the edge itself, the sum of the estimated lengths of the edges 
    descending from this edge's terminal node, and the _average_ length
    of all pendant edges placed on this edge (weighted by the estimated
    probability of each such placement.)
    
    - For a target sequence, the length is the expected value, over 
    all placements, of the pendant edge length.

    Note that if prune is true, node_to_weight will contain only
    nodes with a nonzero estimated probability of being an ancestor
    of target_sequences.

    If prune is false and target_sequences=[], node_to_weight
    will give the subtree length associted with each edge in the 
    bare reference tree, ignoring placed sequences.

    """
    sequence_to_ancestors = {}
    node_to_weight = {}
    placements_by_edge = {} # {edge: [(weight1, pendant1, sequence1), ...] ...}
    nonzero_probability_of_ancestry = set()

    target_sequence_set = set(target_sequences)
    if prune:
        # Don't destructively modify the input...
        tree = tree.copy()
        keep_nodes = set()
        keep_nodes = {edge for seq, options in placements.items() for 
                      _, edge, _, _ in options if seq in target_sequence_set}
        tree.prune(list(keep_nodes), preserve_branch_length=True)

    # Rely on the assumption that all nodes have unique names.
    name_to_node = {n.name: n for n in tree.get_descendants()}
    name_to_node[tree.name] = tree
    for sequence in target_sequences:
        ancestors = {}
        for weight, edge, distal, pendant in placements[sequence]:
            # Note the edge _numbers_ from the placements are converted 
            # to node names as _strings_-- this is taken care of in
            # organize_placements.
            placements_by_edge.setdefault(edge,[]).append((weight, pendant, sequence))
            edge_node = name_to_node[edge]
            for node in [edge_node] + edge_node.get_ancestors():
                nonzero_probability_of_ancestry.add(node.name)
                ancestors[node.name] = weight + ancestors.get(node.name,0.)
        sequence_to_ancestors[sequence] = ancestors

    # walk from leaves back up
    # To avoid confusion with the placement weights, refer to these as
    # lengths, even though they are only sort of lengths.
    for node in reversed(list(tree.traverse(strategy='levelorder'))):
        length = node.dist 
        length += sum(node_to_weight[child.name] for child in node.children)
        if node.name in placements_by_edge:
            weights = np.array([weight for weight, _, _ in
                                placements_by_edge[node.name]])
            pendants = np.array([pendant for _, pendant, _ in
                                 placements_by_edge[node.name]])
            weighted_average_pendant = (np.sum(weights*pendants) /
                                        np.sum(weights)) 
            length += weighted_average_pendant
        node_to_weight[node.name] = length

    # Finally handle the placed sequences.
    for sequence, options in placements.items():
        if sequence not in target_sequences:
            continue
        weights, _, _, pendants = zip(*options)
        weights = np.array(weights)
        pendants = np.array(pendants)
        weighted_average_pendant = np.sum(weights*pendants)/np.sum(weights)
        node_to_weight[sequence] = weighted_average_pendant
            
    return sequence_to_ancestors, node_to_weight, tree


def load_table(placement_filename,
               sequence_key_fasta_filename=None):
    """ Load and return a table of taxonomic placements.

    If needed, use a fasta file to translate sequences to sequence
    IDs in the table index before returning.

    """
    # First get the loading of the sequence key out of the way,
    # where necessary.
    # Recall the RSVs are by definition unique
    sequence_id_map = {}
    if sequence_key_fasta_filename is not None:
        sequence_id_map.update(
             fasta_to_dict(sequence_key_fasta_filename).items()
        )

    # Load the placement table
    placements = pd.read_csv(placement_filename, index_col=0)
    placements = placements.rename(index=sequence_id_map)
    return placements

    
def annotate_dataset_table(dataset, 
                           placement_filename,
                           sequence_key_fasta_filename=None):
    """ Map sequence IDs to sensible taxonomic labels (from table.)

    Here we work with a table of the format typically produced 
    from dada2's RDP-based or RDP-plus-exact-sequence-match-based
    taxonomic placement utilities, specifically a CSV table
    mapping OTUs/RSVs to taxonomic groupings at various levels.

    From this table we assign a succinct description to each 
    OTU/RSV, and describe each higher node in the tree based 
    on the taxonomic groupings of the OTUs which descend from it.

    Note this should be applied at a stage in the filtering
    process when all leaves of the trees correspond to OTUs/RSVs 
    in the original observed data, and vice versa- otherwise
    higher groups will be misidentified as OTUs.

    Annotations are stored in dataset.variable_annotations, a
    dictionary keyed by variable name. It is not updated as the
    dataset is transformed, dropping variables.  (As always, it's the
    user's responsibility to ensure names are unique.)

    dataset: rules.Dataset object 
    node_labels: dictionary mapping node
    placement_filename: csv table of OTU/RSV taxonomies
    sequence_key_fasta_filename: optional fasta file; if given, the
    row labels in the placement file are presumed to be DNA sequences,
    as in DADA2 output, and mapped back to the ids in the fasta where
    possible before return. By default all keys are stripped of
    quotation marks.

    Returns: none.

    """
    placements = load_table(placement_filename,
                            sequence_key_fasta_filename)
    
    # Process to label the OTUs
    otu_labels = {}
    for id_, record in placements.iterrows():
        if record.notnull()['Species']:
            label = '%s %s' % (record['Genus'], record['Species'])
        elif record.notnull().any():
            placed_levels = record[record.notnull()]
            label = '%s %s' % (placed_levels.index[-1],
                               placed_levels.values[-1])
        else:
            label = '(unclassifiable sequence)'
        otu_labels[id_] = 'OTU mapped to %s' % label

    # Process to label the nodes
    nodes = list(dataset.variable_tree.iter_descendants())
    nodes.append(dataset.variable_tree)
    node_labels = {}
    for node in nodes:
        if node.is_leaf():
            continue

        n_leaves = len(node)
        leaf_names = node.get_leaf_names()
        subtable = placements.loc[leaf_names]
        taxa = []
        for level_index, level in enumerate(subtable.columns):
            valid = subtable[level].notnull()
            values = list(set(subtable[level][valid].values))
            if subtable[level].isnull().any():
                values.append('(unclassifiable)')
            taxa.append(values)
            if (len(values) > 1) or (subtable[level].isnull().any()):
                break

        # Case 1: Single species, or species indistinguishable
        if (level.lower() == 'species' and len(taxa[-1]) == 1):
            descriptor = taxa[-2][0] + ' ' + taxa[-1][0]
        # Case 2: At least one consensus level
        # (before a non-consensus level)
        elif len(taxa) > 1:
            consensus_level = subtable.columns[level_index-1].lower()
            split_level = subtable.columns[level_index].lower()
            consensus_value = taxa[level_index-1][0]
            split_values = ', '.join(map(str,taxa[-1]))
            descriptor = ('a clade within %s %s, '
                          'including representatives of %s %s' %
                          (consensus_level, consensus_value,
                           split_level, split_values))
        # Case 3: No consensus levels
        else:
            split_level = subtable.columns[level_index].lower()
            split_values = ', '.join(map(str,taxa[-1]))
            descriptor = ('a clade including representatives of %s %s' %
                          (split_level, split_values))
        node_labels[node.name] = (
            '%s (with %s OTUs [before filtering])' % 
            (descriptor, n_leaves)
        )
    annotations = node_labels.copy()
    annotations.update(otu_labels)
    dataset.variable_annotations = annotations


def annotate_dataset_hybrid(dataset,         
                            jplace_filename,
                            taxa_table_filename,
                            seq_info_filename,
                            placement_filename,
                            sequence_key_fasta_filename=None
                            ):
    """ Add taxonomic information to dataset (from pplacer info plus table).

    Here we use the pplacer reference package to label inner nodes of
    the reference tree. Leaves are labeled according to a table (eg,
    from mothur or dada2's RDP-based or
    RDP-plus-exact-sequence-match-based taxonomic placement utilities)
    if the table contains a placement of the leaf to the species
    level; otherwise, they are labeled according to their parent node
    in the pplacer results.

    In practice, we have found that the dada2 RDP plus sequence
    matching approach often provides more specific placements for many
    OTUs/RSVs than the pplacer approach. This method lets us take
    advantage of that to obtain good annotations while also continuing
    to define inner nodes based on the pplacer reference tree.

    Note this should be applied at a stage in the filtering
    process when all leaves of the trees correspond to OTUs/RSVs 
    in the original observed data, and vice versa- otherwise
    higher groups will be misidentified as OTUs.

    Annotations are stored in dataset.variable_annotations, a
    dictionary keyed by variable name. It is not updated as the
    dataset is transformed, dropping variables.  (As always, it's the
    user's responsibility to ensure names are unique.)

    dataset: rules.Dataset object
    jplace_filename: pplacer output .jplace file 
    taxa_table_filename: pplacer refpkg taxa table
    seq_info_filename: pplacer refpkg seqinfo file
    placement_filename: csv table of OTU/RSV taxonomies
    sequence_key_fasta_filename: optional fasta file; if given, the
    row labels in the placement file are presumed to be DNA sequences,
    as in DADA2 output, and mapped back to the ids in the fasta where
    possible before return. By default all keys are stripped of
    quotation marks.

    Returns: none.

    """
    # First label everything using the pplacer tree.
    # For efficiency, specify which labels we are looking for.
    nodes = list(dataset.variable_tree.iter_descendants())
    nodes.append(dataset.variable_tree)
    node_label_targets = [node.name for node in nodes if not node.is_leaf()]
    
    node_labels = describe_tree_nodes_with_taxonomy(
        jplace_filename,
        taxa_table_filename,
        seq_info_filename,
        to_label = node_label_targets
    )
    # We're going to traverse this tree with no particular
    # regard for inefficiency.
    annotations = {}
    for node in nodes:
        if node.is_leaf():
            parent = node.up.name
            parent_clade = node_labels[parent]
            otu_string = 'OTU mapped to %s' % parent_clade
            annotations[node.name] = otu_string
        else:
            n_leaves = len(node)
            clade = node_labels[node.name]
            annotations[node.name] = (
                '%s' % 
                (clade,)
            )
    # Now read in the table and replace labels
    # for the leaves which are assigned to the species level within the table.

    # First get the loading of the sequence key out of the way,
    # where necessary.
    # Recall the RSVs are by definition unique
    sequence_id_map = {}
    if sequence_key_fasta_filename is not None:
        sequence_id_map.update(
             fasta_to_dict(sequence_key_fasta_filename).items()
        )

    # Load the placement table
    placements = pd.read_csv(placement_filename, index_col=0)
    placements = placements.rename(index=sequence_id_map)
    
    # Grab species-specific OTU labels
    otu_labels = {}
    for id_, record in placements.iterrows():
        if record.notnull()['Species']:
            label = '%s %s' % (record['Genus'], record['Species'])
            otu_labels[id_] = 'OTU mapped to %s' % label
    annotations.update(otu_labels)

    # Finally, save the annotations.
    dataset.variable_annotations = annotations

def log_transform(data,zero_data_offset=1e-6,zero_tolerance=1e-10):
    new_data = data.copy()
    # We expect the data to be positive, so don't take the absolute
    # value before checking to see what is close to zero
    for i in range(len(new_data.X)):
        new_data.X[i][new_data.X[i]<zero_tolerance] = zero_data_offset
        new_data.X[i] = np.log(new_data.X[i])
    return new_data

def discard_low_abundance(data, 
                          min_abundance_threshold, 
                          min_consecutive_samples=2, 
                          min_n_subjects=1,
                          skip_variables=set()):

    """ Drop taxa too rarely above threshold over consecutive time points.

    Specifically, for each variable (except those given by name in 
    skip_variables), we count the number of subjects for which that
    variable exceeds min_abundance_threshold in at least 
    min_consecutive_samples successive time points. If that number
    of subjects is less than min_n_subjects, the variable is 
    dropped. 

    Returns: a new Dataset object and an array of indices of variables
    _kept_ in the filtering process (to allow the same transformation to
    be performed on other data.) 

    A ValueError will be raised if the selected conditions filter out
    all the variables.

    """
    keep_indices = []
    for i in range(data.n_variables):
        if data.variable_names[i] in skip_variables:
            keep_indices.append(i)
            continue
        n_passing_subjects = 0
        for subject_data in data.X:
            above_threshold = subject_data[i] > min_abundance_threshold
            if not any(above_threshold):
                continue
            # is the longest run longer than min_consecutive_samples?
            # This approach is overkill, but foolproof (and, once
            # you know what groupby does, readable:)
            run_lengths = [sum(group) for key,group in
                           itertools.groupby(above_threshold) if
                           key]
            if max(run_lengths) >= min_consecutive_samples:
                n_passing_subjects += 1
        if n_passing_subjects >= min_n_subjects:
            keep_indices.append(i)
    return select_variables(data, keep_indices), keep_indices

def log_transform_if_needed(config, target_data):
    if config.has_option('preprocessing', 'log_transform'):
        if config.getboolean('preprocessing','log_transform'):
            print('Applying log transform...')
            target_data = log_transform(target_data)
    return target_data

def temporal_filter_if_needed(config, target_data):
    if config.has_option('preprocessing','temporal_abundance_threshold'):
        print('Appying temporal filtering')
        target_data, _ = discard_low_abundance(
            target_data,
            min_abundance_threshold = config.getfloat(
                'preprocessing',
                'temporal_abundance_threshold'),
            min_consecutive_samples = config.getfloat(
                'preprocessing',
                'temporal_abundance_consecutive_samples'),
            min_n_subjects = config.getfloat(
                'preprocessing',
                'temporal_abundance_n_subjects')
        )
    return target_data

def discard_surplus_internal_nodes(data):
    """ Drop inner nodes of variable tree not needed to maintain topology.

    All variables which are leaves of the tree, not nodes of the tree 
    at all, or are needed to preserve relationships between the leaves,
    are kept.

    """
    keep_indices = []
    # This list won't include the root, but the root is going to
    # be kept by the pruning process anyway, so it doesn't matter.
    old_tree_nodes = {node.name for node in 
                      data.variable_tree.get_descendants()}
    leaves = data.variable_tree.get_leaf_names()
    new_tree = data.variable_tree.copy()
    new_tree.prune(list(leaves), preserve_branch_length=True)
    new_tree_nodes = {node.name for node in new_tree.get_descendants()}
    print('old/new nodes:')
    print(old_tree_nodes)
    print(new_tree_nodes)
    for i, name in enumerate(data.variable_names):
        print(name)
        if name not in old_tree_nodes:
            print('keep')
            keep_indices.append(i)
        elif name in new_tree_nodes:
            print('keep')
            keep_indices.append(i)

    # Kludge solution for some slightly odd behavior
    # of the tree pruning routine, which keeps more nodes than are in
    # new_tree when prune(new_tree_nodes) is called

    new_data = select_variables(data, keep_indices)
    new_data.variable_tree = new_tree
    return new_data, keep_indices


def write_variable_table(dataset, filename):
    """ Dump notes on every variable to a text file.

    Produces a tab-delimited table, first column the variable name,
    second its annotation in dataset.variable_annotations (if any),
    third column the names of its descendant leaves (if any).

    """ 
    fields = ['description','leaves']
    df = pd.DataFrame(columns=fields, index=dataset.variable_names)
    for name in dataset.variable_names:
        df.loc[name,'description'] = (
            dataset.variable_annotations.get(name, '(no annotation)')
        )
        node_list = dataset.variable_tree.search_nodes(name=name)
        if not node_list:
            leaves_as_string = '(not in variable tree)'
        elif len(node_list) > 1:
            raise ValueError('Ambiguous map from variables to tree')
        else:
            node = node_list[0]
            if node.is_leaf():
                # Leave this field as empty/NA
                continue
            leaves_as_string = ' '.join(node.get_leaf_names())
        df.loc[name,'leaves'] = leaves_as_string
    df.to_csv(filename,sep='\t',index_label='name')
    return df

def preprocess_step2(config, data):
    """ Aggregate, optionally transform, temporal abundance filter, etc.

    Taxonomy information is applied here (after the tree 
    is established; currently no facility for annotating
    taxonomies without a tree; to be added.)
    

    """
    # 3f. Phylogenetic aggregation.
    has_tree = False
    if config.has_option('preprocessing', 'aggregate_on_phylogeny'):
        if config.getboolean('preprocessing','aggregate_on_phylogeny'):
            print('Phylogenetic aggregation begins.')
            jplace_file = config.get('data', 'jplace_file') 
            data, _, _ = aggregate_by_pplacer_simplified(
                jplace_file,
                data
            )
            has_tree = True
            describe_dataset(data,'After phylogenetic aggregation:')
    

    # 3f(b). Optional taxonomy information.
    if has_tree and config.has_option('data','taxonomy_source'):
        # Valid options are 'pplacer' and 'table' and 'hybrid'
        taxonomy_source = config.get('data','taxonomy_source')
        print('Parsing taxonomic annotations.')
    else:
        taxonomy_source = None

    if taxonomy_source == 'table':
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        annotate_dataset_table(
            data,
            placement_table_filename,
            sequence_fasta_filename
        )

    elif taxonomy_source == 'pplacer':
        jplace_filename = config.get('data', 'jplace_file')
        taxa_table_filename = config.get('data', 'pplacer_taxa_table')
        seq_info_filename = config.get('data', 'pplacer_seq_info')
        annotate_dataset_pplacer(
            data,
            jplace_filename,
            taxa_table_filename,
            seq_info_filename
        )

    elif taxonomy_source == 'hybrid':
        jplace_filename = config.get('data', 'jplace_file')
        taxa_table_filename = config.get('data', 'pplacer_taxa_table')
        seq_info_filename = config.get('data', 'pplacer_seq_info')
        placement_table_filename = config.get('data','placement_table')
        if config.has_option('data','sequence_key'):
            sequence_fasta_filename = config.get('data','sequence_key')
        else:
            sequence_fasta_filename = None
        annotate_dataset_hybrid(
            data,
            jplace_filename,
            taxa_table_filename,
            seq_info_filename,
            placement_table_filename,
            sequence_fasta_filename
        )
        
    # 3g. Log transform
    data = log_transform_if_needed(config, data)

    # 3h. Temporal abundance filter.
    # We drop all variables except those which exceed a threshold
    # abundance for a certain number of consecutive observations in a
    # certain number of subjects
    data = temporal_filter_if_needed(config, data)

    # 3i. Surplus internal node removal.
    if has_tree and config.has_option('preprocessing', 'discard_surplus_internal_nodes'):
        if config.getboolean('preprocessing',
                             'discard_surplus_internal_nodes'):
            print('Removing surplus internal nodes...')
            data, _ = discard_surplus_internal_nodes(data)
            describe_dataset(
                data,
                ('After removing internal nodes ' +
                 'not needed to maintain topology:')
            ) 

    # Debugging feature: randomize the labels
    if (config.has_option('preprocessing', 'randomize_labels') and
        config.getboolean('preprocessing', 'randomize_labels')):
        np.random.shuffle(data.y)

    # 3h. Pickling.
    prefix = config.get('description','tag')
    if config.has_option('preprocessing', 'pickle_dataset'):
        if config.getboolean('preprocessing','pickle_dataset'):
            print('Saving dataset...')
            filename = prefix + '_dataset_object.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(data,f)
            print('Dataset written to %s' % filename)

    # 3i. Write taxonomic annotations (if they exist), now
    # that all filtering has been done.
    if taxonomy_source is not None:
        prefix = config.get('description','tag')
        filename = prefix + '_variable_annotations.txt'
        write_variable_table(data,filename)
    return data


if __name__ == '__main__':
    filename = './datasets/raw/t1d/t1d_benchmark.cfg'
    config = ConfigParser.ConfigParser()
    config.read(filename)
    current_dataset = None

    current_dataset = preprocess(config)

    describe_dataset(current_dataset)