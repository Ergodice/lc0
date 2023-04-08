/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "mcts/params.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include "utils/exception.h"
#include "utils/string.h"

#if __has_include("params_override.h")
#include "params_override.h"
#endif

#ifndef DEFAULT_MINIBATCH_SIZE
#define DEFAULT_MINIBATCH_SIZE 256
#endif
#ifndef DEFAULT_TASK_WORKERS
#define DEFAULT_TASK_WORKERS 3
#endif

namespace lczero {

namespace {
FillEmptyHistory EncodeHistoryFill(std::string history_fill) {
  if (history_fill == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (history_fill == "always") return FillEmptyHistory::ALWAYS;
  assert(history_fill == "no");
  return FillEmptyHistory::NO;
}

}  // namespace

const OptionId SearchParams::kMiniBatchSizeId{
    "minibatch-size", "MinibatchSize",
    "How many positions the engine tries to batch together for parallel NN "
    "computation. Larger batches may reduce strength a bit, especially with a "
    "small number of playouts."};
const OptionId SearchParams::kCpuctId{
    "cpuct", "CPuct",
    "cpuct_init constant from \"UCT search\" algorithm. Higher values promote "
    "more exploration/wider search, lower values promote more "
    "confidence/deeper search."};
const OptionId SearchParams::kCpuctAtRootId{
    "cpuct-at-root", "CPuctAtRoot",
    "cpuct_init constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctBaseId{
    "cpuct-base", "CPuctBase",
    "cpuct_base constant from \"UCT search\" algorithm. Lower value means "
    "higher growth of Cpuct as number of node visits grows."};
const OptionId SearchParams::kCpuctBaseAtRootId{
    "cpuct-base-at-root", "CPuctBaseAtRoot",
    "cpuct_base constant from \"UCT search\" algorithm, for root node."};
const OptionId SearchParams::kCpuctFactorId{
    "cpuct-factor", "CPuctFactor", "Multiplier for the cpuct growth formula."};
const OptionId SearchParams::kCpuctFactorAtRootId{
    "cpuct-factor-at-root", "CPuctFactorAtRoot",
    "Multiplier for the cpuct growth formula at root."};
// Remove this option after 0.25 has been made mandatory in training and the
// training server stops sending it.
const OptionId SearchParams::kRootHasOwnCpuctParamsId{
    "root-has-own-cpuct-params", "RootHasOwnCpuctParams",
    "If enabled, cpuct parameters for root node are taken from *AtRoot "
    "parameters. Otherwise, they are the same as for the rest of nodes. "
    "Temporary flag for transition to a new version."};
const OptionId SearchParams::kTwoFoldDrawsId{
    "two-fold-draws", "TwoFoldDraws",
    "Evaluates twofold repetitions in the search tree as draws. Visits to "
    "these positions are reverted when the first occurrence is played "
    "and not in the search tree anymore."};
const OptionId SearchParams::kTemperatureId{
    "temperature", "Temperature",
    "Tau value from softmax formula for the first move. If equal to 0, the "
    "engine picks the best move to make. Larger values increase randomness "
    "while making the move."};
const OptionId SearchParams::kTempDecayMovesId{
    "tempdecay-moves", "TempDecayMoves",
    "Reduce temperature for every move after the first move, decreasing "
    "linearly over this number of moves from initial temperature to 0. "
    "A value of 0 disables tempdecay."};
const OptionId SearchParams::kTempDecayDelayMovesId{
    "tempdecay-delay-moves", "TempDecayDelayMoves",
    "Delay the linear decrease of temperature by this number of moves, "
    "decreasing linearly from initial temperature to 0. A value of 0 starts "
    "tempdecay after the first move."};
const OptionId SearchParams::kTemperatureCutoffMoveId{
    "temp-cutoff-move", "TempCutoffMove",
    "Move number, starting from which endgame temperature is used rather "
    "than initial temperature. Setting it to 0 disables cutoff."};
const OptionId SearchParams::kTemperatureEndgameId{
    "temp-endgame", "TempEndgame",
    "Temperature used during endgame (starting from cutoff move). Endgame "
    "temperature doesn't decay."};
const OptionId SearchParams::kTemperatureWinpctCutoffId{
    "temp-value-cutoff", "TempValueCutoff",
    "When move is selected using temperature, bad moves (with win "
    "probability less than X than the best move) are not considered at all."};
const OptionId SearchParams::kTemperatureVisitOffsetId{
    "temp-visit-offset", "TempVisitOffset",
    "Adjusts visits by this value when picking a move with a temperature. If a "
    "negative offset reduces visits for a particular move below zero, that "
    "move is not picked. If no moves can be picked, no temperature is used."};
const OptionId SearchParams::kNoiseEpsilonId{
    "noise-epsilon", "DirichletNoiseEpsilon",
    "Amount of Dirichlet noise to combine with root priors. This allows the "
    "engine to discover new ideas during training by exploring moves which are "
    "known to be bad. Not normally used during play."};
const OptionId SearchParams::kNoiseAlphaId{
    "noise-alpha", "DirichletNoiseAlpha",
    "Alpha of Dirichlet noise to control the sharpness of move probabilities. "
    "Larger values result in flatter / more evenly distributed values."};
const OptionId SearchParams::kVerboseStatsId{
    "verbose-move-stats", "VerboseMoveStats",
    "Display Q, V, N, U and P values of every move candidate after each move.",
    'v'};
const OptionId SearchParams::kLogLiveStatsId{
    "log-live-stats", "LogLiveStats",
    "Do VerboseMoveStats on every info update."};
const OptionId SearchParams::kFpuStrategyId{
    "fpu-strategy", "FpuStrategy",
    "How is an eval of unvisited node determined. \"First Play Urgency\" "
    "changes search behavior to visit unvisited nodes earlier or later by "
    "using a placeholder eval before checking the network. The value specified "
    "with --fpu-value results in \"reduction\" subtracting that value from the "
    "parent eval while \"absolute\" directly uses that value."};
const OptionId SearchParams::kFpuValueId{
    "fpu-value", "FpuValue",
    "\"First Play Urgency\" value used to adjust unvisited node eval based on "
    "--fpu-strategy."};
const OptionId SearchParams::kFpuStrategyAtRootId{
    "fpu-strategy-at-root", "FpuStrategyAtRoot",
    "How is an eval of unvisited root children determined. Just like "
    "--fpu-strategy except only at the root level and adjusts unvisited root "
    "children eval with --fpu-value-at-root. In addition to matching the "
    "strategies from --fpu-strategy, this can be \"same\" to disable the "
    "special root behavior."};
const OptionId SearchParams::kFpuValueAtRootId{
    "fpu-value-at-root", "FpuValueAtRoot",
    "\"First Play Urgency\" value used to adjust unvisited root children eval "
    "based on --fpu-strategy-at-root. Has no effect if --fpu-strategy-at-root "
    "is \"same\"."};
const OptionId SearchParams::kCacheHistoryLengthId{
    "cache-history-length", "CacheHistoryLength",
    "Length of history, in half-moves, to include into the cache key. When "
    "this value is less than history that NN uses to eval a position, it's "
    "possble that the search will use eval of the same position with different "
    "history taken from cache."};
const OptionId SearchParams::kPolicySoftmaxTempId{
    "policy-softmax-temp", "PolicyTemperature",
    "Policy softmax temperature. Higher values make priors of move candidates "
    "closer to each other, widening the search."};
const OptionId SearchParams::kMaxCollisionVisitsId{
    "max-collision-visits", "MaxCollisionVisits",
    "Total allowed node collision visits, per batch."};
const OptionId SearchParams::kMaxCollisionEventsId{
    "max-collision-events", "MaxCollisionEvents",
    "Allowed node collision events, per batch."};
const OptionId SearchParams::kOutOfOrderEvalId{
    "out-of-order-eval", "OutOfOrderEval",
    "During the gathering of a batch for NN to eval, if position happens to be "
    "in the cache or is terminal, evaluate it right away without sending the "
    "batch to the NN. When off, this may only happen with the very first node "
    "of a batch; when on, this can happen with any node."};
const OptionId SearchParams::kMaxOutOfOrderEvalsId{
    "max-out-of-order-evals-factor", "MaxOutOfOrderEvalsFactor",
    "Maximum number of out of order evals during gathering of a batch is "
    "calculated by multiplying the maximum batch size by this number."};
const OptionId SearchParams::kStickyEndgamesId{
    "sticky-endgames", "StickyEndgames",
    "When an end of game position is found during search, allow the eval of "
    "the previous move's position to stick to something more accurate. For "
    "example, if at least one move results in checkmate, then the position "
    "should stick as checkmated. Similarly, if all moves are drawn or "
    "checkmated, the position should stick as drawn or checkmate."};
const OptionId SearchParams::kSyzygyFastPlayId{
    "syzygy-fast-play", "SyzygyFastPlay",
    "With DTZ tablebase files, only allow the network pick from winning moves "
    "that have shortest DTZ to play faster (but not necessarily optimally)."};
const OptionId SearchParams::kMultiPvId{
    "multipv", "MultiPV",
    "Number of game play lines (principal variations) to show in UCI info "
    "output."};
const OptionId SearchParams::kPerPvCountersId{
    "per-pv-counters", "PerPVCounters",
    "Show node counts per principal variation instead of total nodes in UCI."};
const OptionId SearchParams::kScoreTypeId{
    "score-type", "ScoreType",
    "What to display as score. Either centipawns (the UCI default), win "
    "percentage or Q (the actual internal score) multiplied by 100."};
const OptionId SearchParams::kHistoryFillId{
    "history-fill", "HistoryFill",
    "Neural network uses 7 previous board positions in addition to the current "
    "one. During the first moves of the game such historical positions don't "
    "exist, but they can be synthesized. This parameter defines when to "
    "synthesize them (always, never, or only at non-standard fen position)."};
const OptionId SearchParams::kMovesLeftMaxEffectId{
    "moves-left-max-effect", "MovesLeftMaxEffect",
    "Maximum bonus to add to the score of a node based on how much "
    "shorter/longer it makes the game when winning/losing."};
const OptionId SearchParams::kMovesLeftThresholdId{
    "moves-left-threshold", "MovesLeftThreshold",
    "Absolute value of node Q needs to exceed this value before shorter wins "
    "or longer losses are considered."};
const OptionId SearchParams::kMovesLeftSlopeId{
    "moves-left-slope", "MovesLeftSlope",
    "Controls how the bonus for shorter wins or longer losses is adjusted "
    "based on how many moves the move is estimated to shorten/lengthen the "
    "game. The move difference is multiplied with the slope and capped at "
    "MovesLeftMaxEffect."};
const OptionId SearchParams::kMovesLeftConstantFactorId{
    "moves-left-constant-factor", "MovesLeftConstantFactor",
    "A simple multiplier to the moves left effect, can be set to 0 to only use "
    "an effect scaled by Q."};
const OptionId SearchParams::kMovesLeftScaledFactorId{
    "moves-left-scaled-factor", "MovesLeftScaledFactor",
    "A factor which is multiplied by the absolute Q of parent node and the "
    "base moves left effect."};
const OptionId SearchParams::kMovesLeftQuadraticFactorId{
    "moves-left-quadratic-factor", "MovesLeftQuadraticFactor",
    "A factor which is multiplied by the square of Q of parent node and the "
    "base moves left effect."};
const OptionId SearchParams::kDisplayCacheUsageId{
    "display-cache-usage", "DisplayCacheUsage",
    "Display cache fullness through UCI info `hash` section."};
const OptionId SearchParams::kMaxConcurrentSearchersId{
    "max-concurrent-searchers", "MaxConcurrentSearchers",
    "If not 0, at most this many search workers can be gathering minibatches "
    "at once."};
const OptionId SearchParams::kDrawScoreSidetomoveId{
    "draw-score-sidetomove", "DrawScoreSideToMove",
    "Score of a drawn game, as seen by a player making the move."};
const OptionId SearchParams::kDrawScoreOpponentId{
    "draw-score-opponent", "DrawScoreOpponent",
    "Score of a drawn game, as seen by the opponent."};
const OptionId SearchParams::kDrawScoreWhiteId{
    "draw-score-white", "DrawScoreWhite",
    "Adjustment, added to a draw score of a white player."};
const OptionId SearchParams::kDrawScoreBlackId{
    "draw-score-black", "DrawScoreBlack",
    "Adjustment, added to a draw score of a black player."};
const OptionId SearchParams::kContemptPerspectiveId{
    "contempt-perspective", "ContemptPerspective",
    "Affects the way asymmetric WDL parameters are applied. Default is "
    "'sidetomove' for matches, use 'white' and 'black' for analysis. Use "
    "'none' to deactivate contempt and the WDL conversion."};
const OptionId SearchParams::kContemptId{
    "contempt", "Contempt",
    "The simulated rating advantage for the WDL conversion. Comma separated "
    "list in the form [name=]value, where the name is compared with the "
    "`UCI_Opponent` value to find the appropriate contempt value. The default "
    "value is taken from `UCI_RatingAdv` and will be overridden if either a "
    "value without name is given, or if a name match is found."};
const OptionId SearchParams::kContemptMaxValueId{
    "contempt-max-value", "ContemptMaxValue",
    "The maximum value of contempt used. Higher values will be capped."};
const OptionId SearchParams::kWDLRescaleRatioId{
    "wdl-rescale-ratio", "WDLRescaleRatio",
    "Rescales the logistic WDL scale by the given ratio."};
const OptionId SearchParams::kWDLRescaleDiffId{
    "wdl-rescale-diff", "WDLRescaleDiff",
    "Shifts the logistic WDL mean by diff in white's favor."};
const OptionId SearchParams::kWDLCalibrationEloId{
    "wdl-calibration-elo", "WDLCalibrationElo",
    "Elo of the active side, adjusted for time control relative to rapid."};
const OptionId SearchParams::kWDLContemptAttenuationId{
    "wdl-contempt-attenuation", "WDLContemptAttenuation",
    "Scales how Elo advantage is applied for contempt. Use 1.0 for realistic "
    "analysis, and 0.5-0.6 for optimal match performance."};
const OptionId SearchParams::kWDLEvalObjectivityId{
    "wdl-eval-objectivity", "WDLEvalObjectivity",
    "When calculating the centipawn eval output, decides how objective/"
    "contempt influenced the reported eval should be. Value 0.0 reports the "
    "internally used WDL values, 1.0 attempts an objective eval."};
const OptionId SearchParams::kWDLDrawRateTargetId{
    "wdl-draw-rate-target", "WDLDrawRateTarget",
    "To define the accuracy of play, the target draw rate in equal "
    "positions is used as a proxy."};
const OptionId SearchParams::kWDLDrawRateReferenceId{
    "wdl-draw-rate-reference", "WDLDrawRateReference",
    "Set this to the draw rate predicted by the used neural network at "
    "default settings. The accuracy rescaling is done relative to the "
    "reference draw rate."};
const OptionId SearchParams::kWDLBookExitBiasId{
    "wdl-book-exit-bias", "WDLBookExitBias",
    "The book exit bias used when measuring engine Elo. Value of startpos is "
    "around 0.2, value of 50% white win is 1. Only relevant if target draw "
    "rate is above 80%."};
const OptionId SearchParams::kNpsLimitId{
    "nps-limit", "NodesPerSecondLimit",
    "An option to specify an upper limit to the nodes per second searched. The "
    "accuracy depends on the minibatch size used, increasing for lower sizes, "
    "and on the length of the search. Zero to disable."};
const OptionId SearchParams::kTaskWorkersPerSearchWorkerId{
    "task-workers", "TaskWorkers",
    "The number of task workers to use to help the search worker."};
const OptionId SearchParams::kMinimumWorkSizeForProcessingId{
    "minimum-processing-work", "MinimumProcessingWork",
    "This many visits need to be gathered before tasks will be used to "
    "accelerate processing."};
const OptionId SearchParams::kMinimumWorkSizeForPickingId{
    "minimum-picking-work", "MinimumPickingWork",
    "Search branches with more than this many collisions/visits may be split "
    "off to task workers."};
const OptionId SearchParams::kMinimumRemainingWorkSizeForPickingId{
    "minimum-remaining-picking-work", "MinimumRemainingPickingWork",
    "Search branches won't be split off to task workers unless there is at "
    "least this much work left to do afterwards."};
const OptionId SearchParams::kMinimumWorkPerTaskForProcessingId{
    "minimum-per-task-processing", "MinimumPerTaskProcessing",
    "Processing work won't be split into chunks smaller than this (unless its "
    "more than half of MinimumProcessingWork)."};
const OptionId SearchParams::kIdlingMinimumWorkId{
    "idling-minimum-work", "IdlingMinimumWork",
    "Only early exit gathering due to 'idle' backend if more than this many "
    "nodes will be sent to the backend."};
const OptionId SearchParams::kThreadIdlingThresholdId{
    "thread-idling-threshold", "ThreadIdlingThreshold",
    "If there are more than this number of search threads that are not "
    "actively in the process of either sending data to the backend or waiting "
    "for data from the backend, assume that the backend is idle."};
const OptionId SearchParams::kMaxCollisionVisitsScalingStartId{
    "max-collision-visits-scaling-start", "MaxCollisionVisitsScalingStart",
    "Tree size where max collision visits starts scaling up from 1."};
const OptionId SearchParams::kMaxCollisionVisitsScalingEndId{
    "max-collision-visits-scaling-end", "MaxCollisionVisitsScalingEnd",
    "Tree size where max collision visits reaches max. Set to 0 to disable "
    "scaling entirely."};
const OptionId SearchParams::kMaxCollisionVisitsScalingPowerId{
    "max-collision-visits-scaling-power", "MaxCollisionVisitsScalingPower",
    "Power to apply to the interpolation between 1 and max to make it curved."};
const OptionId SearchParams::kUCIOpponentId{
    "", "UCI_Opponent",
    "UCI option used by the GUI to pass the name and other information about "
    "the current opponent."};
const OptionId SearchParams::kUCIRatingAdvId{
    "", "UCI_RatingAdv",
    "UCI extension used by some GUIs to pass the estimated Elo difference from "
    "the current opponent, used as the default contempt value."};

void SearchParams::Populate(OptionsParser* options) {
  // Here the uci optimized defaults" are set.
  // Many of them are overridden with training specific values in tournament.cc.
  options->Add<IntOption>(kMiniBatchSizeId, 1, 1024) = DEFAULT_MINIBATCH_SIZE;
  options->Add<FloatOption>(kCpuctId, 0.0f, 100.0f) = 1.745f;
  options->Add<FloatOption>(kCpuctAtRootId, 0.0f, 100.0f) = 1.745f;
  options->Add<FloatOption>(kCpuctBaseId, 1.0f, 1000000000.0f) = 38739.0f;
  options->Add<FloatOption>(kCpuctBaseAtRootId, 1.0f, 1000000000.0f) = 38739.0f;
  options->Add<FloatOption>(kCpuctFactorId, 0.0f, 1000.0f) = 3.894f;
  options->Add<FloatOption>(kCpuctFactorAtRootId, 0.0f, 1000.0f) = 3.894f;
  options->Add<BoolOption>(kRootHasOwnCpuctParamsId) = false;
  options->Add<BoolOption>(kTwoFoldDrawsId) = true;
  options->Add<FloatOption>(kTemperatureId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kTempDecayMovesId, 0, 640) = 0;
  options->Add<IntOption>(kTempDecayDelayMovesId, 0, 100) = 0;
  options->Add<IntOption>(kTemperatureCutoffMoveId, 0, 1000) = 0;
  options->Add<FloatOption>(kTemperatureEndgameId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kTemperatureWinpctCutoffId, 0.0f, 100.0f) = 100.0f;
  options->Add<FloatOption>(kTemperatureVisitOffsetId, -1000.0f, 1000.0f) =
      0.0f;
  options->Add<FloatOption>(kNoiseEpsilonId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kNoiseAlphaId, 0.0f, 10000000.0f) = 0.3f;
  options->Add<BoolOption>(kVerboseStatsId) = false;
  options->Add<BoolOption>(kLogLiveStatsId) = false;
  std::vector<std::string> fpu_strategy = {"reduction", "absolute"};
  options->Add<ChoiceOption>(kFpuStrategyId, fpu_strategy) = "reduction";
  options->Add<FloatOption>(kFpuValueId, -100.0f, 100.0f) = 0.330f;
  fpu_strategy.push_back("same");
  options->Add<ChoiceOption>(kFpuStrategyAtRootId, fpu_strategy) = "same";
  options->Add<FloatOption>(kFpuValueAtRootId, -100.0f, 100.0f) = 1.0f;
  options->Add<IntOption>(kCacheHistoryLengthId, 0, 7) = 0;
  options->Add<FloatOption>(kPolicySoftmaxTempId, 0.1f, 10.0f) = 1.359f;
  options->Add<IntOption>(kMaxCollisionEventsId, 1, 65536) = 917;
  options->Add<IntOption>(kMaxCollisionVisitsId, 1, 100000000) = 80000;
  options->Add<IntOption>(kMaxCollisionVisitsScalingStartId, 1, 100000) = 28;
  options->Add<IntOption>(kMaxCollisionVisitsScalingEndId, 0, 100000000) =
      145000;
  options->Add<FloatOption>(kMaxCollisionVisitsScalingPowerId, 0.01, 100) =
      1.25;
  options->Add<BoolOption>(kOutOfOrderEvalId) = true;
  options->Add<FloatOption>(kMaxOutOfOrderEvalsId, 0.0f, 100.0f) = 2.4f;
  options->Add<BoolOption>(kStickyEndgamesId) = true;
  options->Add<BoolOption>(kSyzygyFastPlayId) = false;
  options->Add<IntOption>(kMultiPvId, 1, 500) = 1;
  options->Add<BoolOption>(kPerPvCountersId) = false;
  std::vector<std::string> score_type = {"centipawn",
                                         "centipawn_with_drawscore",
                                         "centipawn_2019",
                                         "centipawn_2018",
                                         "win_percentage",
                                         "Q",
                                         "W-L",
                                         "WDL_mu"};
  options->Add<ChoiceOption>(kScoreTypeId, score_type) = "centipawn";
  std::vector<std::string> history_fill_opt{"no", "fen_only", "always"};
  options->Add<ChoiceOption>(kHistoryFillId, history_fill_opt) = "fen_only";
  options->Add<FloatOption>(kMovesLeftMaxEffectId, 0.0f, 1.0f) = 0.0345f;
  options->Add<FloatOption>(kMovesLeftThresholdId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kMovesLeftSlopeId, 0.0f, 1.0f) = 0.00f;
  options->Add<FloatOption>(kMovesLeftConstantFactorId, -1.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kMovesLeftScaledFactorId, -2.0f, 2.0f) = 1.6521f;
  options->Add<FloatOption>(kMovesLeftQuadraticFactorId, -1.0f, 1.0f) =
      -0.6521f;
  options->Add<BoolOption>(kDisplayCacheUsageId) = false;
  options->Add<IntOption>(kMaxConcurrentSearchersId, 0, 128) = 1;
  options->Add<IntOption>(kDrawScoreSidetomoveId, -100, 100) = 0;
  options->Add<IntOption>(kDrawScoreOpponentId, -100, 100) = 0;
  options->Add<IntOption>(kDrawScoreWhiteId, -100, 100) = 0;
  options->Add<IntOption>(kDrawScoreBlackId, -100, 100) = 0;
  std::vector<std::string> perspective = {"sidetomove", "white", "black",
                                          "none"};
  options->Add<ChoiceOption>(kContemptPerspectiveId, perspective) =
      "sidetomove";
  options->Add<StringOption>(kContemptId) = "";
  options->Add<FloatOption>(kContemptMaxValueId, 0, 10000.0f) = 420.0f;
  options->Add<FloatOption>(kWDLRescaleRatioId, 1e-6f, 1e6f) = 1.0f;
  options->Add<FloatOption>(kWDLRescaleDiffId, -100.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kWDLCalibrationEloId, 0, 10000.0f) = 0.0f;
  options->Add<FloatOption>(kWDLContemptAttenuationId, -10.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kWDLEvalObjectivityId, 0.0f, 1.0f) = 1.0f;
  options->Add<FloatOption>(kWDLDrawRateTargetId, 0.001f, 0.999f) = 0.5f;
  options->Add<FloatOption>(kWDLDrawRateReferenceId, 0.001f, 0.999f) = 0.5f;
  options->Add<FloatOption>(kWDLBookExitBiasId, -2.0f, 2.0f) = 0.65f;
  options->Add<FloatOption>(kNpsLimitId, 0.0f, 1e6f) = 0.0f;
  options->Add<IntOption>(kTaskWorkersPerSearchWorkerId, 0, 128) =
      DEFAULT_TASK_WORKERS;
  options->Add<IntOption>(kMinimumWorkSizeForProcessingId, 2, 100000) = 20;
  options->Add<IntOption>(kMinimumWorkSizeForPickingId, 1, 100000) = 1;
  options->Add<IntOption>(kMinimumRemainingWorkSizeForPickingId, 0, 100000) =
      20;
  options->Add<IntOption>(kMinimumWorkPerTaskForProcessingId, 1, 100000) = 8;
  options->Add<IntOption>(kIdlingMinimumWorkId, 0, 10000) = 0;
  options->Add<IntOption>(kThreadIdlingThresholdId, 0, 128) = 1;
  options->Add<StringOption>(kUCIOpponentId);
  options->Add<FloatOption>(kUCIRatingAdvId, -10000.0f, 10000.0f) = 0.0f;

  options->HideOption(kNoiseEpsilonId);
  options->HideOption(kNoiseAlphaId);
  options->HideOption(kLogLiveStatsId);
  options->HideOption(kDisplayCacheUsageId);
  options->HideOption(kRootHasOwnCpuctParamsId);
  options->HideOption(kCpuctAtRootId);
  options->HideOption(kCpuctBaseAtRootId);
  options->HideOption(kCpuctFactorAtRootId);
  options->HideOption(kFpuStrategyAtRootId);
  options->HideOption(kFpuValueAtRootId);
  options->HideOption(kTemperatureId);
  options->HideOption(kTempDecayMovesId);
  options->HideOption(kTempDecayDelayMovesId);
  options->HideOption(kTemperatureCutoffMoveId);
  options->HideOption(kTemperatureEndgameId);
  options->HideOption(kTemperatureWinpctCutoffId);
  options->HideOption(kTemperatureVisitOffsetId);
  options->HideOption(kContemptMaxValueId);
  options->HideOption(kWDLRescaleRatioId);
  options->HideOption(kWDLRescaleDiffId);
  options->HideOption(kWDLContemptAttenuationId);
  options->HideOption(kWDLDrawRateTargetId);
  options->HideOption(kWDLBookExitBiasId);
}

SearchParams::SearchParams(const OptionsDict& options)
    : options_(options),
      kCpuct(options.Get<float>(kCpuctId)),
      kCpuctAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctAtRootId
                                                      : kCpuctId)),
      kCpuctBase(options.Get<float>(kCpuctBaseId)),
      kCpuctBaseAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctBaseAtRootId
                                                      : kCpuctBaseId)),
      kCpuctFactor(options.Get<float>(kCpuctFactorId)),
      kCpuctFactorAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctFactorAtRootId
                                                      : kCpuctFactorId)),
      kTwoFoldDraws(options.Get<bool>(kTwoFoldDrawsId)),
      kNoiseEpsilon(options.Get<float>(kNoiseEpsilonId)),
      kNoiseAlpha(options.Get<float>(kNoiseAlphaId)),
      kFpuAbsolute(options.Get<std::string>(kFpuStrategyId) == "absolute"),
      kFpuValue(options.Get<float>(kFpuValueId)),
      kFpuAbsoluteAtRoot(
          (options.Get<std::string>(kFpuStrategyAtRootId) == "same" &&
           kFpuAbsolute) ||
          options.Get<std::string>(kFpuStrategyAtRootId) == "absolute"),
      kFpuValueAtRoot(options.Get<std::string>(kFpuStrategyAtRootId) == "same"
                          ? kFpuValue
                          : options.Get<float>(kFpuValueAtRootId)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthId)),
      kPolicySoftmaxTemp(options.Get<float>(kPolicySoftmaxTempId)),
      kMaxCollisionEvents(options.Get<int>(kMaxCollisionEventsId)),
      kMaxCollisionVisits(options.Get<int>(kMaxCollisionVisitsId)),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalId)),
      kStickyEndgames(options.Get<bool>(kStickyEndgamesId)),
      kSyzygyFastPlay(options.Get<bool>(kSyzygyFastPlayId)),
      kHistoryFill(EncodeHistoryFill(options.Get<std::string>(kHistoryFillId))),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeId)),
      kMovesLeftMaxEffect(options.Get<float>(kMovesLeftMaxEffectId)),
      kMovesLeftThreshold(options.Get<float>(kMovesLeftThresholdId)),
      kMovesLeftSlope(options.Get<float>(kMovesLeftSlopeId)),
      kMovesLeftConstantFactor(options.Get<float>(kMovesLeftConstantFactorId)),
      kMovesLeftScaledFactor(options.Get<float>(kMovesLeftScaledFactorId)),
      kMovesLeftQuadraticFactor(
          options.Get<float>(kMovesLeftQuadraticFactorId)),
      kDisplayCacheUsage(options.Get<bool>(kDisplayCacheUsageId)),
      kMaxConcurrentSearchers(options.Get<int>(kMaxConcurrentSearchersId)),
      kDrawScoreSidetomove{options.Get<int>(kDrawScoreSidetomoveId) / 100.0f},
      kDrawScoreOpponent{options.Get<int>(kDrawScoreOpponentId) / 100.0f},
      kDrawScoreWhite{options.Get<int>(kDrawScoreWhiteId) / 100.0f},
      kDrawScoreBlack{options.Get<int>(kDrawScoreBlackId) / 100.0f},
      kContemptPerspective(options.Get<std::string>(kContemptPerspectiveId)),
      kWDLEvalObjectivity(options.Get<float>(kWDLEvalObjectivityId)),
      kMaxOutOfOrderEvals(std::max(
          1, static_cast<int>(options.Get<float>(kMaxOutOfOrderEvalsId) *
                              options.Get<int>(kMiniBatchSizeId)))),
      kNpsLimit(options.Get<float>(kNpsLimitId)),
      kTaskWorkersPerSearchWorker(
          options.Get<int>(kTaskWorkersPerSearchWorkerId)),
      kMinimumWorkSizeForProcessing(
          options.Get<int>(kMinimumWorkSizeForProcessingId)),
      kMinimumWorkSizeForPicking(
          options.Get<int>(kMinimumWorkSizeForPickingId)),
      kMinimumRemainingWorkSizeForPicking(
          options.Get<int>(kMinimumRemainingWorkSizeForPickingId)),
      kMinimumWorkPerTaskForProcessing(
          options.Get<int>(kMinimumWorkPerTaskForProcessingId)),
      kIdlingMinimumWork(options.Get<int>(kIdlingMinimumWorkId)),
      kThreadIdlingThreshold(options.Get<int>(kThreadIdlingThresholdId)),
      kMaxCollisionVisitsScalingStart(
          options.Get<int>(kMaxCollisionVisitsScalingStartId)),
      kMaxCollisionVisitsScalingEnd(
          options.Get<int>(kMaxCollisionVisitsScalingEndId)),
      kMaxCollisionVisitsScalingPower(
          options.Get<float>(kMaxCollisionVisitsScalingPowerId)) {
  // Calculate ratio and diff for WDL conversion from the contempt settings,
  // unless they are changed from their default values explicitly.
  kWDLRescaleRatio = options.Get<float>(kWDLRescaleRatioId);
  kWDLRescaleDiff = options.Get<float>(kWDLRescaleDiffId);
  if (kWDLRescaleRatio == 1.0 && kWDLRescaleDiff == 0.0) {
    // The default kWDLContemptId is empty, so the initial contempt value is
    // taken from kUCIRatingAdvId. Adding any value (without name) in the comma
    // separated kWDLContemptId list will override this.
    float contempt = options.Get<float>(kUCIRatingAdvId);
    if (!options.IsDefault<std::string>(kContemptId)) {
      auto name = options.Get<std::string>(kUCIOpponentId);
      for (auto& entry : StrSplit(options.Get<std::string>(kContemptId), ",")) {
        auto parts = StrSplit(entry, "=");
        if (parts.size() == 1) {
          try {
            contempt = std::stof(parts[0]);
          } catch (std::exception& e) {
            throw Exception("Invalid default contempt: " + entry);
          }
        } else if (parts.size() == 2) {
          if (std::search(name.begin(), name.end(), parts[0].begin(),
                          parts[0].end(), [](unsigned char a, unsigned char b) {
                            return std::tolower(a) == std::tolower(b);
                          }) != name.end()) {
            try {
              contempt = std::stof(parts[1]);
            } catch (std::exception& e) {
              throw Exception("Invalid contempt entry: " + entry);
            }
            break;
          }
        } else {
          throw Exception("Invalid contempt entry:" + entry);
        }
      }
    }

    if (options.Get<float>(kWDLCalibrationEloId) == 0) {
      // More accurate model, allowing book bias dependent Elo calculation.
      // Doesn't take lower accuracy of opponent into account and needs
      // clamping.
      float scale_target =
          1.0f / std::log((1.0f + options.Get<float>(kWDLDrawRateTargetId)) /
                          (1.0f - options.Get<float>(kWDLDrawRateTargetId)));
      float scale_reference =
          1.0f / std::log((1.0f + options.Get<float>(kWDLDrawRateReferenceId)) /
                          (1.0f - options.Get<float>(kWDLDrawRateReferenceId)));
      kWDLRescaleRatio = scale_target / scale_reference;
      kWDLRescaleDiff =
          scale_target / (scale_reference * scale_reference) /
          (1.0f /
               std::pow(std::cosh(0.5f *
                                  (1 - options.Get<float>(kWDLBookExitBiasId)) /
                                  scale_target),
                        2) +
           1.0f /
               std::pow(std::cosh(0.5f *
                                  (1 + options.Get<float>(kWDLBookExitBiasId)) /
                                  scale_target),
                        2)) *
          std::log(10) / 200 *
          std::clamp(contempt, -options.Get<float>(kContemptMaxValueId),
                     options.Get<float>(kContemptMaxValueId)) *
          options.Get<float>(kWDLContemptAttenuationId);
    } else {
      // Less accurate Elo model, but automatically chooses draw rate and
      // accuracy based on the absolute Elo of both sides. Doesn't require
      // clamping, but still uses the parameter.
      // Parameters for the Elo dependent draw rate and scaling:
      const float scale_zero = 15.0f;
      const float elo_slope = 425.0f;
      const float offset = 6.75f;

      float scale_reference =
          1.0f / std::log((1.0f + options.Get<float>(kWDLDrawRateReferenceId)) /
                          (1.0f - options.Get<float>(kWDLDrawRateReferenceId)));
      float elo_active = options.Get<float>(kWDLCalibrationEloId);
      float elo_opp =
          elo_active - std::clamp(contempt,
                                  -options.Get<float>(kContemptMaxValueId),
                                  options.Get<float>(kContemptMaxValueId));
      float scale_active = 1.0f / (1.0f / scale_zero +
                                   std::exp(elo_active / elo_slope - offset));
      float scale_opp =
          1.0f / (1.0f / scale_zero + std::exp(elo_opp / elo_slope - offset));
      float scale_target = std::sqrt(
          (scale_active * scale_active + scale_opp * scale_opp) / 2.0f);
      kWDLRescaleRatio = scale_target / scale_reference;
      float mu_active =
          -std::log(10) / 200 * scale_zero * elo_slope *
          std::log(1.0f +
                   std::exp(-elo_active / elo_slope + offset) / scale_zero);
      float mu_opp =
          -std::log(10) / 200 * scale_zero * elo_slope *
          std::log(1.0f + std::exp(-elo_opp / elo_slope + offset) / scale_zero);
      kWDLRescaleDiff =
          (mu_active - mu_opp) * options.Get<float>(kWDLContemptAttenuationId);
    }
  }
  if (std::max(std::abs(kDrawScoreSidetomove), std::abs(kDrawScoreOpponent)) +
          std::max(std::abs(kDrawScoreWhite), std::abs(kDrawScoreBlack)) >
      1.0f) {
    throw Exception(
        "max{|sidetomove|+|opponent|} + max{|white|+|black|} draw score must "
        "be <= 100");
  }
}

}  // namespace lczero
