% =========================================================================
% STANDALONE SCRIPT TO GENERATE STATS.MAT FILES FROM RAW EXPERIMENTAL DATA
% =========================================================================
%
% PURPOSE:
% This script processes a single ephys/behavioral experiment to create a
% '_Stats.mat' file. It is a generalized version of the core logic from
% your automated processing pipeline.
%
% HOW TO USE:
% 1. Save this entire file as 'generate_stats_file.m'.
% 2. Make sure the folder containing this file is on your MATLAB path,
%    or that you are in the correct directory in MATLAB.
% 3. Ensure all required lab-specific functions (like 'loadData',
%    'Project_Selectivity_computeResp_perImage', etc.) are also on the path.
% 4. Call the main function from the MATLAB command window or another
%    script using the format shown in the example below.
%
% =========================================================================
%% ========================================================================
%                              USAGE EXAMPLE
% =========================================================================
%{
% --- To run this, uncomment the block and execute it in MATLAB ---
% Add paths to your lab's function repositories if they are not already set
% addpath('C:\Users\rickysilva\Documents\MATLAB\Ponce-Lab-Functions\'); % Example path
% Define the experiments you want to process in a cell array
experiments_to_process = {
    {'ephysFN', 'Caos-06092025-007', 'controlFN', '250609_141516_Caos_visual_search', 'prefChan', [76 1], 'stimuli', "N:\PonceLab\Stimuli\Invariance\Project_Invariance\Stimuli_backups_from_experiments\09_06_25\Search2", 'type', 'search', 'monkey', 'Caos'},...
    {'ephysFN', 'Caos-06092025-009', 'controlFN', '250609_144326_Caos_visual_search', 'prefChan', [76 1], 'stimuli', "N:\PonceLab\Stimuli\Invariance\Project_Invariance\Stimuli_backups_from_experiments\09_06_25\Search2", 'type', 'search', 'monkey', 'Caos'},...
    {'ephysFN', 'Caos-06092025-010', 'controlFN', '250609_144937_Caos_visual_search', 'prefChan', [76 1], 'stimuli', "N:\PonceLab\Stimuli\Invariance\Project_Invariance\Stimuli_backups_from_experiments\09_06_25\Search3", 'type', 'search', 'monkey', 'Caos'}...
};
% Loop through and process each experiment
for i = 1:length(experiments_to_process)
    fprintf('\n================== PROCESSING EXPERIMENT %d of %d ==================\n', i, length(experiments_to_process));
    % Convert the cell array for the current experiment into a struct
    exp_info = struct(experiments_to_process{i}{:});
    % Call the main processing function
    generate_stats_file(exp_info);
end
%}
%% ========================================================================
%                            MAIN FUNCTION
% =========================================================================
function [success, reason] = generate_stats_file(exp_info)
    % This is the main function that processes a single experiment.
    % INPUT:
    %   exp_info: A structure with all necessary fields for processing.
    %             Required fields: ephysFN, controlFN, prefChan, stimuli, type, monkey
    %
    % OUTPUT:
    %   success: True if processing was successful, false otherwise.
    %   reason: A string describing the outcome ('processed', 'skipped', 'failed').
    %% --- Configuration and Setup ---
    source_ephys = 'N:\PonceLab\Data-Ephys-Raw';
    source_behavior = 'N:\PonceLab\Data-Behavior (BHV2)';
    dest_ephys = 'S:\Data-Ephys-Raw';
    dest_behavior = 'S:\Data-Behaviour';
    path_to_save = 'S:\Data-Ephys-MAT';
    % Define analysis windows
    windows.early = 1:50;
    windows.late = 60:200;
    % Initialize log files (optional but good practice)
    error_log_file = fullfile(path_to_save, 'general_processing_errors.log');
    alignment_log_file = fullfile(path_to_save, 'general_alignment_diagnostics.log');
    success_log_file = fullfile(path_to_save, 'general_processing_success.log');
    if ~exist(error_log_file, 'file')
        initializeLogs(error_log_file, alignment_log_file, success_log_file);
    end
    % --- Function Execution ---
    [success, reason] = processSingleExperiment(exp_info, ...
        source_ephys, source_behavior, dest_ephys, dest_behavior, path_to_save, windows, ...
        error_log_file, alignment_log_file, success_log_file);
end
%% ========================================================================
%                      CORE PROCESSING LOGIC (HELPER)
% =========================================================================
function [success, reason] = processSingleExperiment(exp_info, ...
    source_ephys, source_behavior, dest_ephys, dest_behavior, path_to_save, ...
    windows, error_log_file, alignment_log_file, success_log_file)
    % Extract info from the input struct for clarity
    exp_name = exp_info.ephysFN;
    control_name = exp_info.controlFN;
    success = false;
    reason = 'unknown';
    fprintf('    Processing: %s + %s\n', exp_name, control_name);
    % Check if already processed
    output_file = fullfile(path_to_save, [exp_name '_Stats.mat']);
    if exist(output_file, 'file')
        fprintf('    %s already processed, skipping...\n', exp_name);
        logSuccess(success_log_file, exp_name, control_name, 'Already processed');
        success = true;
        reason = 'already_processed';
        return;
    end
    copied_files = {};
    try
        %% Step 1: Copy files to local drive
        fprintf('    Copying files to local drive...\n');
        % Copy ephys file (.pl2)
        source_ephys_file = fullfile(source_ephys, [exp_name '.pl2']);
        dest_ephys_file = fullfile(dest_ephys, [exp_name '.pl2']);
        if exist(source_ephys_file, 'file')
            copyfile(source_ephys_file, dest_ephys_file);
            copied_files{end+1} = dest_ephys_file;
        else
            error('Ephys file not found: %s', source_ephys_file);
        end
        % Copy behavioral file (.bhv2)
        source_bhv_file = fullfile(source_behavior, [control_name '.bhv2']);
        dest_bhv_file = fullfile(dest_behavior, [control_name '.bhv2']);
        if exist(source_bhv_file, 'file')
            copyfile(source_bhv_file, dest_bhv_file);
            copied_files{end+1} = dest_bhv_file;
        else
            error('Behavioral file not found: %s', source_bhv_file);
        end
        %% Step 2: Check for truncated recordings
        pl2_file = fullfile(dest_ephys, [exp_name '.pl2']);
        [~, evts] = plx_event_ts(pl2_file, 257); % Word channel
        if isempty(evts)
            error('No words found in ephys file. Cannot check duration.');
        end
        recording_duration = max(evts) - min(evts);
        fprintf('      Recording duration: %.1f minutes\n', recording_duration/60);
        if recording_duration < 15 % Exclude recordings shorter than 15 seconds
            error('Ephys recording truncated (%.1f sec)', recording_duration);
        end
        %% Step 3: Process the experiment
        fprintf('    Processing experiment...\n');
        % Create metadata structure
        tMeta = exp_info;
        % Load data using your lab's function
        fprintf('      Loading ephys and behavioral data...\n');
        % NOTE: Assumes 'loadData' is a function on your MATLAB path.
        [meta_, rasters, ~, Trials] = loadData(tMeta.ephysFN, 'expControlFN', tMeta.controlFN);
        % Merge metadata
        fprintf('      Merging metadata...\n');
        meta = mergeMetadataSafely(tMeta, meta_);
        % Convert to single precision to save memory
        rasters = single(rasters);
        % Create main data structure
        M.rasters{1} = rasters;
        M.meta{1} = meta;
        M.Trials{1} = Trials;
        M.monkey = exp_info.monkey;
        % Compute responses per image
        fprintf('      Computing responses per image...\n');
        % NOTE: Assumes 'Project_Selectivity_computeResp_perImage' is on your path.
        Stats_tmp = Project_Selectivity_computeResp_perImage({meta}, {Trials}, {rasters}, 'windows', windows);
        % Add additional metadata to the final Stats structure
        Stats_tmp{1}.type = exp_info.type;
        if isfield(exp_info, 'phase'), Stats_tmp{1}.phase = exp_info.phase; end
        if isfield(exp_info, 'search_session'), Stats_tmp{1}.search_session = exp_info.search_session; end
        if isfield(exp_info, 'stimuli') && ~isempty(exp_info.stimuli)
            Stats_tmp{1}.stimuli = exp_info.stimuli;
        end
        M.Stats = Stats_tmp;
        % Save results
        fprintf('      Saving results...\n');
        % NOTE: Assumes 'Project_Scramble_saveExperimentsWithInfoAndPath' is on your path.
        Project_Scramble_saveExperimentsWithInfoAndPath(M);
        % Log success
        logSuccess(success_log_file, exp_name, control_name, sprintf('Successfully processed %d trials', length(Trials)));
        fprintf('    :white_tick: Successfully processed %s\n', exp_name);
        success = true;
        reason = 'processed';
    catch ME
        fprintf('    :x: ERROR processing %s: %s\n', exp_name, ME.message);
        logError(error_log_file, exp_name, control_name, ME.getReport());
        reason = 'failed';
    end
    %% Step 4: Clean up local files
    fprintf('    Cleaning up local files...\n');
    cleanupFiles(copied_files);
end
%% ========================================================================
%                      ALL OTHER HELPER FUNCTIONS
% =========================================================================
function meta = mergeMetadataSafely(tMeta, meta_)
    % Safely merge metadata structures, handling overlapping fields
    try
        overlap_fields = intersect(fieldnames(tMeta), fieldnames(meta_));
        if ~isempty(overlap_fields)
            tMeta_clean = rmfield(tMeta, overlap_fields);
        else
            tMeta_clean = tMeta;
        end
        names = [fieldnames(tMeta_clean); fieldnames(meta_)];
        meta = cell2struct([struct2cell(tMeta_clean); struct2cell(meta_)], names, 1);
    catch ME
        warning('Metadata merge failed, using basic merge: %s', ME.message);
        meta = meta_;
        essential_fields = {'type', 'phase', 'search_session', 'prefChan'};
        for i = 1:length(essential_fields)
            field = essential_fields{i};
            if isfield(tMeta, field)
                meta.(field) = tMeta.(field);
            end
        end
    end
end
function cleanupFiles(copied_files)
    % Clean up the files that were copied for this experiment
    for iFile = 1:length(copied_files)
        if exist(copied_files{iFile}, 'file')
            delete(copied_files{iFile});
        end
    end
end
function logError(log_file, exp_name, control_name, error_msg)
    % Log errors to file
    fid = fopen(log_file, 'a');
    fprintf(fid, '%s: ERROR - %s + %s - %s\n', datestr(now), exp_name, control_name, error_msg);
    fclose(fid);
end
function logSuccess(log_file, exp_name, control_name, success_msg)
    % Log successful processing to file
    fid = fopen(log_file, 'a');
    fprintf(fid, '%s: SUCCESS - %s + %s - %s\n', datestr(now), exp_name, control_name, success_msg);
    fclose(fid);
end
function initializeLogs(error_log_file, alignment_log_file, success_log_file)
    % Initialize log files with headers
    fid = fopen(error_log_file, 'w');
    fprintf(fid, '=== PROCESSING ERROR LOG - %s ===\n', datestr(now));
    fclose(fid);
    fid = fopen(alignment_log_file, 'w');
    fprintf(fid, '=== ALIGNMENT DIAGNOSTIC LOG - %s ===\n', datestr(now));
    fclose(fid);
    fid = fopen(success_log_file, 'w');
    fprintf(fid, '=== PROCESSING SUCCESS LOG - %s ===\n', datestr(now));
    fclose(fid);
end






