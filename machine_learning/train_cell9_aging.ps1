param(
    [int]$Epochs = 100,
    [string]$DataDirectory = (Join-Path $env:USERPROFILE 'Downloads'),
    [string]$Output = 'outputs/real_cell9_0_200',
    [ValidateSet('lstm', 'mlp')][string]$Model = 'lstm',
    [switch]$MlpSearch,
    [string]$CompareLstmCheckpoint = ''
)

if ($Model -eq 'mlp' -and -not $PSBoundParameters.ContainsKey('Output')) {
    $Output = 'outputs/mlp_cell9_0_200'
}

# Run from the repository root. Inputs: Cell 9, 25 C, at 0 and 200 cycles.
# Fresh capacity has '(1)' in its name; fresh incremental does not.
$trainingArguments = @(
    '-3.13', '-u', '-m', 'machine_learning.train_real_soh',
    '--input',
    (Join-Path $DataDirectory 'SAMSUNG_Cell_9_incremental_1C_Channel_3_Wb_1.CSV'),
    (Join-Path $DataDirectory 'SAMSUNG_Cell_9_incremental_1C_Channel_3_Wb_1(1).CSV'),
    '--capacity-csv',
    (Join-Path $DataDirectory 'SAMSUNG_Cell_9_capacity_01C_Channel_3_Wb_1(1).CSV'),
    (Join-Path $DataDirectory 'SAMSUNG_Cell_9_capacity_01C_Channel_3_Wb_1.CSV'),
    '--bol-capacity-csv',
    (Join-Path $DataDirectory 'SAMSUNG_Cell_9_capacity_01C_Channel_3_Wb_1(1).CSV'),
    '--epochs', $Epochs,
    '--output', $Output
)
$trainingArguments += @('--model', $Model)
if ($MlpSearch) { $trainingArguments += '--mlp-search' }
if ($CompareLstmCheckpoint) { $trainingArguments += @('--compare-lstm-checkpoint', $CompareLstmCheckpoint) }
& py @trainingArguments
exit $LASTEXITCODE
