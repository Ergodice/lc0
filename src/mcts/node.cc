/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "mcts/node.h"

#include <absl/algorithm/container.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_set>

#include "utils/exception.h"
#include "utils/hashcat.h"

namespace lczero {





/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Mirror();
  return m;
}

// Policy priors (P) are stored in a compressed 16-bit format.
//
// Source values are 32-bit floats:
// * bit 31 is sign (zero means positive)
// * bit 30 is sign of exponent (zero means nonpositive)
// * bits 29..23 are value bits of exponent
// * bits 22..0 are significand bits (plus a "virtual" always-on bit: s ∈ [1,2))
// The number is then sign * 2^exponent * significand, usually.
// See https://www.h-schmidt.net/FloatConverter/IEEE754.html for details.
//
// In compressed 16-bit value we store bits 27..12:
// * bit 31 is always off as values are always >= 0
// * bit 30 is always off as values are always < 2
// * bits 29..28 are only off for values < 4.6566e-10, assume they are always on
// * bits 11..0 are for higher precision, they are dropped leaving only 11 bits
//     of precision
//
// When converting to compressed format, bit 11 is added to in order to make it
// a rounding rather than truncation.
//
// Out of 65556 possible values, 2047 are outside of [0,1] interval (they are in
// interval (1,2)). This is fine because the values in [0,1] are skewed towards
// 0, which is also exactly how the components of policy tend to behave (since
// they add up to 1).

// If the two assumed-on exponent bits (3<<28) are in fact off, the input is
// rounded up to the smallest value with them on. We accomplish this by
// subtracting the two bits from the input and checking for a negative result
// (the subtraction works despite crossing from exponent to significand). This
// is combined with the round-to-nearest addition (1<<11) into one op.
void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.as_string() << " p_: " << p_ << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// LowNode + Node
/////////////////////////////////////////////////////////////////////////

// Put @low_node at the end of TT @gc_queue, if both @gc_queue and @low_node
// are not null and &low_node is TT and about to become parent-less (has only
// one parent).
static void TTGCEnqueue(GCQueue* gc_queue, const LowNode* low_node) {
  if (gc_queue && low_node && low_node->IsTT() &&
      low_node->GetNumParents() == 1)
    gc_queue->push_back(low_node->GetHash());
}

void Node::Trim(GCQueue* gc_queue) {
  wl_ = 0.0f;
  score_ = wl_;

  TTGCEnqueue(gc_queue, low_node_);
  UnsetLowNode();
  // sibling_

  d_ = 0.0f;
  m_ = 0.0f;
  vs_ = 0.0f;
  n_ = 0;
  weight_ = 0.0f;
  n_in_flight_ = 0;

  // edge_

  // index_

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  repetition_ = false;
}

Node* Node::GetChild() const {
  if (!low_node_) return nullptr;
  return low_node_->GetChild()->get();
}

bool Node::HasChildren() const { return low_node_ && low_node_->HasChildren(); }

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += node->GetP();
  return sum;
}

uint32_t Node::GetNInFlight() const {
  return n_in_flight_.load(std::memory_order_acquire);
}

uint32_t Node::GetChildrenVisits() const {
  return low_node_ ? low_node_->GetChildrenVisits() : 0;
}

uint32_t Node::GetTotalVisits() const {
  return low_node_ ? low_node_->GetN() : 0;
}
float Node::GetV() const {
  return low_node_ ? -low_node_->GetV() : 0.0f;
}

float Node::GetCHDelta() const { return low_node_ ? -low_node_->GetCHDelta() : 0.0f; }

uint64_t Node::GetCHHash() const {
  return low_node_ ? low_node_->GetCHHash() : 0;
}

uint64_t Node::GetHash() const {
  return low_node_ ? low_node_->GetHash() : 0;
}


const Edge& LowNode::GetEdgeAt(uint16_t index) const { return edges_[index]; }

std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " <Node> This:" << this << " LowNode:" << low_node_
      << " Index:" << index_ << " Move:" << GetMove().as_string()
      << " Sibling:" << sibling_.get() << " P:" << GetP() << " WL:" << wl_
      << " D:" << d_ << " M:" << m_ << " N:" << n_ << " N_:" << n_in_flight_
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2;
  return oss.str();
}

std::string LowNode::DebugString() const {
  std::ostringstream oss;
  oss << " <LowNode> This:" << this << " Hash:" << hash_
      << " Edges:" << edges_.get()
      << " NumEdges:" << static_cast<int>(num_edges_)
      << " Child:" << child_.get() << " WL:" << wl_ << " D:" << d_
      << " M:" << m_ << " N:" << n_ << " NP:" << num_parents_
      << " Term:" << static_cast<int>(terminal_type_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << " IsTransposition:" << is_transposition;
  return oss.str();
}

void Edge::SortEdges(Edge* edges, int num_edges) {
  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges, (edges + num_edges),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}

void LowNode::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
  }
  vs_ = wl_ * wl_;
  score_ = wl_;

  assert(WLDMInvariantsHold());
}

void LowNode::MakeNotTerminal(const Node* node) {
  assert(edges_);
  if (!IsTerminal()) return;

  terminal_type_ = Terminal::NonTerminal;
  lower_bound_ = GameResult::BLACK_WON;
  upper_bound_ = GameResult::WHITE_WON;
  n_ = 0;
  weight_ = 0.0;
  wl_ = 0.0;
  score_ = wl_;
  d_ = 0.0;
  m_ = 0.0;
  vs_ = 0.0;

  // Include children too.
  if (node->GetNumEdges() > 0) {
    for (const auto& child : node->Edges()) {
      const auto n = child.GetN();
      const auto weight = child.GetWeight();
      if (n > 0) {
        n_ += n;
        weight_ += weight;
        // Flip Q for opponent.
        // Default values don't matter as n is > 0.
        wl_ += child.GetWL(0.0f) * weight;
        d_ += child.GetD(0.0f) * weight;
        m_ += child.GetM(0.0f) * weight;
        vs_ += child.GetVS(0.0f) * weight;
        score_ += child.GetScore(0.0f) * weight;
      }
    }

    // Recompute with current eval (instead of network's) and children's eval.
    wl_ /= weight_;
    d_ /= weight_;
    m_ /= weight_;
    vs_ /= weight_;
    score_ /= weight_;
  }

  assert(WLDMInvariantsHold());
}

void LowNode::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

uint8_t Node::GetNumEdges() const {
  return low_node_ ? low_node_->GetNumEdges() : 0;
}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    // Terminal losses have no uncertainty and no reason for their U value to be
    // comparable to another non-loss choice. Force this by clearing the policy.
    SetP(0.0f);
  }
  vs_ = wl_ * wl_;
  score_ = wl_;

  assert(WLDMInvariantsHold());
}

void Node::MakeNotTerminal(bool also_low_node) {
  // At least one of node and low node pair needs to be a terminal.
  if (!IsTerminal() &&
      (!also_low_node || !low_node_ || !low_node_->IsTerminal()))
    return;

  terminal_type_ = Terminal::NonTerminal;
  repetition_ = false;
  if (low_node_) {  // Two-fold or derived terminal.
    // Revert low node first.
    if (also_low_node && low_node_) low_node_->MakeNotTerminal(this);

    auto [lower_bound, upper_bound] = low_node_->GetBounds();
    lower_bound_ = -upper_bound;
    upper_bound_ = -lower_bound;
    n_ = low_node_->GetN();
    weight_ = low_node_->GetWeight();
    wl_ = -low_node_->GetWL();
    score_ = -low_node_->GetScore();
    d_ = low_node_->GetD();
    m_ = low_node_->GetM() + 1;
    vs_ = low_node_->GetVS();
  } else {  // Real terminal.
    lower_bound_ = GameResult::BLACK_WON;
    upper_bound_ = GameResult::WHITE_WON;
    n_ = 0.0f;
    weight_ = 0.0f;
    wl_ = 0.0f;
    score_ = wl_;
    d_ = 0.0f;
    m_ = 0.0f;
    vs_ = 0.0f;
  }

  assert(WLDMInvariantsHold());
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (n_ > 0) {
    n_in_flight_.fetch_add(1, std::memory_order_acq_rel);
  } else {
    uint32_t expected_n_if_flight_ = 0;
    if (!n_in_flight_.compare_exchange_strong(expected_n_if_flight_, 1,
                                              std::memory_order_acq_rel)) {
      return false;
    }
  }

  return true;
}

void Node::CancelScoreUpdate(uint32_t multivisit) {
  assert(GetNInFlight() >= (uint32_t)multivisit);
  n_in_flight_.fetch_sub(multivisit, std::memory_order_acq_rel);
}

void LowNode::FinalizeScoreUpdate(float v, float d, float m, float vs, float score,
                                  uint32_t multivisit, float multiweight) {
  assert(edges_);


    
  if (cht_entry_ != nullptr) {
    cht_entry_->weightSum += multiweight;
    cht_entry_->deltaSum += multiweight * (v_ - v);

    ch_delta_ = (cht_entry_->weightSum > 0)
                    ? cht_entry_->deltaSum / cht_entry_->weightSum
                    : 0.0f;
  }

  // Recompute Q.
  wl_ += multiweight * (v - wl_) / (weight_ + multiweight);
  score_ += multiweight * (score - score_) / (weight_ + multiweight);
  d_ += multiweight * (d - d_) / (weight_ + multiweight);
  m_ += multiweight * (m - m_) / (weight_ + multiweight);
  vs_ += multiweight * (vs - vs_) / (weight_ + multiweight);



  assert(WLDMInvariantsHold());

  // Increment N.
  n_ += multivisit;
  weight_ += multiweight;
}


void LowNode::AdjustForTerminal(float v, float d, float m, float vs, float score,
                                uint32_t multivisit, float multiweight) {
  assert(static_cast<uint32_t>(multivisit) <= n_);



  // Recompute Q.
  wl_ += multiweight * v / weight_;
  score_ += multiweight * score / weight_;
  d_ += multiweight * d / weight_;
  m_ += multiweight * m / weight_;
  vs_ += multiweight * vs / weight_;

  if (cht_entry_ != nullptr ) {cht_entry_->deltaSum -= multiweight * v;
  }



  assert(WLDMInvariantsHold());
}



void Node::FinalizeScoreUpdate(float v, float d, float m, float vs, float score,
                               uint32_t multivisit, float multiweight) {




  // Recompute Q.
  wl_ += multiweight * (v - wl_) / (weight_ + multiweight);
  score_ += multiweight * (score - score_) / (weight_ + multiweight);
  d_ += multiweight * (d - d_) / (weight_ + multiweight);
  m_ += multiweight * (m - m_) / (weight_ + multiweight);
  vs_ += multiweight * (vs - vs_) / (weight_ + multiweight);

  assert(WLDMInvariantsHold());

  // Increment N.
  n_ += multivisit;
  weight_ += multiweight;

  // Decrement virtual loss.
  assert(GetNInFlight() >= (uint32_t)multivisit);
  n_in_flight_.fetch_sub(multivisit, std::memory_order_acq_rel);
}

void Node::AdjustForTerminal(float v, float d, float m, float vs, float score,
                             uint32_t multivisit, float multiweight) {
  assert(static_cast<uint32_t>(multivisit) <= n_);

  // Recompute Q.
  wl_ += multiweight * v / weight_;
  score_ += multiweight * score / weight_;
  d_ += multiweight * d / weight_;
  m_ += multiweight * m / weight_;
  vs_ += multiweight * vs / weight_;


  assert(WLDMInvariantsHold());
}

void Node::IncrementNInFlight(uint32_t multivisit) {
  n_in_flight_.fetch_add(multivisit, std::memory_order_acq_rel);
}

void Node::SetE(float e) { e_ = e; }

void LowNode::ReleaseChildren(GCQueue* gc_queue) {
  for (auto child = GetChild()->get(); child != nullptr;
       child = child->GetSibling()->get()) {
    TTGCEnqueue(gc_queue, child->GetLowNode());
  }
  child_.reset();
}

void LowNode::ReleaseChildrenExceptOne(Node* node_to_save, GCQueue* gc_queue) {
  // Stores node which will have to survive (or nullptr if it's not found).
  atomic_unique_ptr<Node> saved_node;
  // Pointer to unique_ptr, so that we could move from it.
  for (auto node = &child_; *node != nullptr; node = (*node)->GetSibling()) {
    // If current node is the one that we have to save.
    if (node->get() == node_to_save) {
      // Save the node, and take the ownership from the unique_ptr.
      saved_node = std::move(*node);
      node = &saved_node;
    } else {
      TTGCEnqueue(gc_queue, (*node)->GetLowNode());
    }
  }
  // Kill all remaining siblings.
  saved_node->GetSibling()->reset();
  // Make saved node the only child. (kills previous siblings).
  child_ = std::move(saved_node);
}

void Node::ReleaseChildrenExceptOne(Node* node_to_save,
                                    GCQueue* gc_queue) const {
  // Sometime we have no graph yet or a reverted terminal without low node.
  if (low_node_) low_node_->ReleaseChildrenExceptOne(node_to_save, gc_queue);
}

void Node::SetLowNode(LowNode* low_node) {
  assert(!low_node_);
  low_node->AddParent();
  low_node_ = low_node;
}
void Node::UnsetLowNode() {
  if (low_node_) low_node_->RemoveParent();
  low_node_ = nullptr;
}

static std::string PtrToNodeName(const void* ptr) {
  std::ostringstream oss;
  oss << "n_" << ptr;
  return oss.str();
}

std::string LowNode::DotNodeString() const {
  std::ostringstream oss;
  oss << PtrToNodeName(this) << " ["
      << "shape=box";
  // Adjust formatting to limit node size.
  oss << std::fixed << std::setprecision(3);
  oss << ",label=\""     //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\lD=" << d_ << "\\lM=" << m_ << "\\lN=" << n_ << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::showpos << std::setprecision(5);
  oss << ",tooltip=\""   //
      << std::showpos    //
      << "WL=" << wl_    //
      << std::noshowpos  //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << n_
      << "\\nNP=" << num_parents_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2
      << "\\nIsTransposition=" << is_transposition  //
      << std::noshowpos                             //
      << "\\n\\nThis=" << this << "\\nEdges=" << edges_.get()
      << "\\nNumEdges=" << static_cast<int>(num_edges_)
      << "\\nChild=" << child_.get() << "\\n\"";
  oss << "];";
  return oss.str();
}

std::string Node::DotEdgeString(bool as_opponent, const LowNode* parent) const {
  std::ostringstream oss;
  oss << (parent == nullptr ? "top" : PtrToNodeName(parent)) << " -> "
      << (low_node_ ? PtrToNodeName(low_node_) : PtrToNodeName(this)) << " [";
  oss << "label=\""
      << (parent == nullptr ? "N/A" : GetMove(as_opponent).as_string())
      << "\\lN=" << n_ << "\\lN_=" << n_in_flight_;
  oss << "\\l\"";
  // Set precision for tooltip.
  oss << std::fixed << std::setprecision(5);
  oss << ",labeltooltip=\""
      << "P=" << (parent == nullptr ? 0.0f : GetP())  //
      << std::showpos                                 //
      << "\\nWL= " << wl_                             //
      << std::noshowpos                               //
      << "\\nD=" << d_ << "\\nM=" << m_ << "\\nN=" << n_
      << "\\nN_=" << n_in_flight_
      << "\\nTerm=" << static_cast<int>(terminal_type_)  //
      << std::showpos                                    //
      << "\\nBounds=" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << "\\n\\nThis=" << this  //
      << std::noshowpos                                               //
      << "\\nLowNode=" << low_node_ << "\\nParent=" << parent
      << "\\nIndex=" << index_ << "\\nSibling=" << sibling_.get() << "\\n\"";
  oss << "];";
  return oss.str();
}

std::string Node::DotGraphString(bool as_opponent) const {
  std::ostringstream oss;
  std::unordered_set<const LowNode*> seen;
  std::list<std::pair<const Node*, bool>> unvisited_fifo;

  oss << "strict digraph {" << std::endl;
  oss << "edge ["
      << "headport=n"
      << ",tooltip=\" \""  // Remove default tooltips from edge parts.
      << "];" << std::endl;
  oss << "node ["
      << "shape=point"    // For fake nodes.
      << ",style=filled"  // Show tooltip everywhere on the node.
      << ",fillcolor=ivory"
      << "];" << std::endl;
  oss << "ranksep=" << 4.0f * std::log10(GetN()) << std::endl;

  oss << DotEdgeString(!as_opponent) << std::endl;
  if (low_node_) {
    seen.insert(low_node_);
    unvisited_fifo.push_back(std::pair(this, as_opponent));
  }

  while (!unvisited_fifo.empty()) {
    auto [parent_node, parent_as_opponent] = unvisited_fifo.front();
    unvisited_fifo.pop_front();

    auto parent_low_node = parent_node->GetLowNode();
    seen.insert(parent_low_node);
    oss << parent_low_node->DotNodeString() << std::endl;

    for (auto& child_edge : parent_node->Edges()) {
      auto child = child_edge.node();
      if (child == nullptr) break;

      oss << child->DotEdgeString(parent_as_opponent) << std::endl;
      auto child_low_node = child->GetLowNode();
      if (child_low_node != nullptr &&
          (seen.find(child_low_node) == seen.end())) {
        seen.insert(child_low_node);
        unvisited_fifo.push_back(std::pair(child, !parent_as_opponent));
      }
    }
  }

  oss << "}" << std::endl;

  return oss.str();
}

bool Node::ZeroNInFlight() const {
  std::unordered_set<const LowNode*> seen;
  std::list<const Node*> unvisited_fifo;
  size_t nonzero_node_count = 0;

  if (GetNInFlight() > 0) {
    std::cerr << DebugString() << std::endl;
    ++nonzero_node_count;
  }
  if (low_node_) {
    seen.insert(low_node_);
    unvisited_fifo.push_back(this);
  }

  while (!unvisited_fifo.empty()) {
    auto parent_node = unvisited_fifo.front();
    unvisited_fifo.pop_front();

    for (auto& child_edge : parent_node->Edges()) {
      auto child = child_edge.node();
      if (child == nullptr) break;

      if (child->GetNInFlight() > 0) {
        std::cerr << child->DebugString() << std::endl;
        ++nonzero_node_count;
      }

      auto child_low_node = child->GetLowNode();
      if (child_low_node != nullptr &&
          (seen.find(child_low_node) == seen.end())) {
        seen.insert(child_low_node);
        unvisited_fifo.push_back(child);
      }
    }
  }

  if (nonzero_node_count > 0) {
    std::cerr << "GetNInFlight() is nonzero on " << nonzero_node_count
              << " nodes" << std::endl;
    return false;
  }

  return true;
}

void Node::SortEdges() const {
  assert(low_node_);
  low_node_->SortEdges();
}


bool Node::IsTT() const { return low_node_ && low_node_->IsTT(); }

static constexpr float wld_tolerance = 0.000001f;
static constexpr float m_tolerance = 0.000001f;


#if 0 // Correction history breaks WLDM invariants

  static bool WLDMInvariantsHold(float wl, float d, float m) {
    return -(1.0f + wld_tolerance) < wl && wl < (1.0f + wld_tolerance) &&  //
           -(0.0f + wld_tolerance) < d && d < (1.0f + wld_tolerance) &&    //
           -(0.0f + m_tolerance) < m &&                                    //
           std::abs(wl + d) < (1.0f + wld_tolerance);
  }

#else

  static bool WLDMInvariantsHold(float wl, float d, float m) {
		return true;
	} 

#endif

bool Node::WLDMInvariantsHold() const {
  if (lczero::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

bool LowNode::WLDMInvariantsHold() const {
  if (lczero::WLDMInvariantsHold(GetWL(), GetD(), GetM())) return true;

  std::cerr << DebugString() << std::endl;

  return false;
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree
/////////////////////////////////////////////////////////////////////////

void NodeTree::MakeMove(Move move) {
  if (HeadPosition().IsBlackToMove()) move.Mirror();
  const auto& board = HeadPosition().GetBoard();
  auto hash = GetHistoryHash(history_);
  move = board.GetModernMove(move);  // TODO: Why convert here?

  // Find edge for @move, if it exists.
  Node* new_head = nullptr;
  while (new_head == nullptr) {
    for (auto& n : current_head_->Edges()) {
      if (board.IsSameMove(n.GetMove(), move)) {
        new_head = n.GetOrSpawnNode(current_head_);
        // Ensure head is not terminal, so search can extend or visit children
        // of "terminal" positions, e.g., WDL hits, converted terminals, 3-fold
        // draw.
        if (new_head->IsTerminal()) new_head->MakeNotTerminal();
        break;
      }
    }

    if (new_head != nullptr) break;

    // Current head node (if any) is non-TT, does not have a matching edge and
    // will be removed by NonTTMaintenance later.
    current_head_->UnsetLowNode();

    // Check TT first, then create, if necessary.
    auto tt_iter = tt_.find(hash);
    if (tt_iter != tt_.end()) {
      current_head_->SetLowNode(tt_iter->second.get());
      if (current_head_->IsTerminal()) current_head_->MakeNotTerminal();
    } else {
      non_tt_.emplace_back(std::make_unique<LowNode>(hash, MoveList({move}),
                                                     static_cast<uint16_t>(0)));
      current_head_->SetLowNode(non_tt_.back().get());
    }
  }

  // Remove edges that will not be needed any more.
  current_head_->ReleaseChildrenExceptOne(new_head, &gc_queue_);
  new_head = current_head_->GetChild();

  // Move damaged node from TT to non-TT to avoid reuse.
  // It can have TT parents, until they get garbage collected.
  if (current_head_->IsTT()) {
    auto tt_iter = tt_.find(current_head_->GetHash());
    tt_iter->second->ClearTT();
    non_tt_.emplace_back(std::move(tt_iter->second));
    tt_.erase(tt_iter);
  }

  current_head_ = new_head;

  history_.Append(move);
  moves_.push_back(move);
}

void NodeTree::TrimTreeAtHead() {
  current_head_->Trim(&gc_queue_);
  // Free unused non-TT low nodes.
  NonTTMaintenance();
}

bool NodeTree::ResetToPosition(const std::string& starting_fen,
                               const std::vector<Move>& moves) {
  ChessBoard starting_board;
  int no_capture_ply;
  int full_moves;
  starting_board.SetFromFen(starting_fen, &no_capture_ply, &full_moves);
  if (gamebegin_node_ &&
      (history_.Starting().GetBoard() != starting_board ||
       history_.Starting().GetRule50Ply() != no_capture_ply)) {
    // Completely different position.
    DeallocateTree();
  }

  if (!gamebegin_node_) {
    gamebegin_node_ = std::make_unique<Node>(0);
  }

  history_.Reset(starting_board, no_capture_ply,
                 full_moves * 2 - (starting_board.flipped() ? 1 : 2));
  moves_.clear();

  Node* old_head = current_head_;
  current_head_ = gamebegin_node_.get();
  bool seen_old_head = (gamebegin_node_.get() == old_head);
  for (const auto& move : moves) {
    MakeMove(move);
    if (old_head == current_head_) seen_old_head = true;
  }

  // Remove any non-TT nodes that were not reused.
  NonTTMaintenance();

  // MakeMove guarantees that no siblings exist; but, if we didn't see the old
  // head, it means we might have a position that was an ancestor to a
  // previously searched position, which means that the current_head_ might
  // retain old n_ and q_ (etc) data, even though its old children were
  // previously trimmed; we need to reset current_head_ in that case.
  if (!seen_old_head) TrimTreeAtHead();
  return seen_old_head;
}

void NodeTree::DeallocateTree() {
  gamebegin_node_.reset();
  current_head_ = nullptr;
  // Free all nodes.
  // There may be non-TT children of TT nodes that were not garbage collected
  // fast enough.
  NonTTMaintenance();
  TTClear();
  non_tt_.clear();
  gc_queue_.clear();
}

LowNode* NodeTree::TTFind(uint64_t hash) {
  auto tt_iter = tt_.find(hash);
  if (tt_iter != tt_.end()) {
    return tt_iter->second.get();
  } else {
    return nullptr;
  }
}

CorrHistEntry* NodeTree::CHTGetOrCreate(uint64_t hash) {
  auto [cht_iter, is_cht_miss] = cht_.insert({hash, std::make_unique<CorrHistEntry>()});
  return cht_iter->second.get();
}

std::pair<LowNode*, bool> NodeTree::TTGetOrCreate(uint64_t hash) {
  auto [tt_iter, is_tt_miss] =
      tt_.insert({hash, std::make_unique<LowNode>(hash)});
  return {tt_iter->second.get(), is_tt_miss};
}

std::pair<LowNode*, bool> NodeTree::TTGetOrCreate(const LowNode& p, uint64_t hash) {
  auto [tt_iter, is_tt_miss] =
      tt_.insert({hash, std::make_unique<LowNode>(p, hash)});
  return {tt_iter->second.get(), is_tt_miss};
}

void NodeTree::TTMaintenance() { TTGCSome(0); }

void NodeTree::TTClear() {
  // Make sure destructors don't fail.
  absl::c_for_each(
      tt_, [](const auto& item) { item.second->ReleaseChildren(nullptr); });
  // Remove any released non-TT children of TT nodes that were not garbage
  // collected fast enough.
  NonTTMaintenance();
  tt_.clear();
  gc_queue_.clear();
}

LowNode* NodeTree::NonTTAddClone(const LowNode& node) {
  non_tt_.push_back(std::make_unique<LowNode>(node));
  return non_tt_.back().get();
}

void NodeTree::NonTTMaintenance() {
  // Release children of parent-less nodes.
  absl::c_for_each(non_tt_, [this](const auto& item) {
    if (item->GetNumParents() == 0) item->ReleaseChildren(&gc_queue_);
  });
  // Erase parent-less nodes.
  for (auto item = non_tt_.begin(); item != non_tt_.end();) {
    if ((*item)->GetNumParents() == 0) {
      item = non_tt_.erase(item);
    } else {
      ++item;
    }
  }
}

bool NodeTree::TTGCSome(size_t count) {
  if (gc_queue_.empty()) return false;

  for (auto n = count > 0 ? std::min(count, gc_queue_.size())
                          : gc_queue_.size();
       n > 0; --n) {
    auto hash = gc_queue_.front();
    gc_queue_.pop_front();
    auto tt_iter = tt_.find(hash);
    if (tt_iter != tt_.end()) {
      if (tt_iter->second->GetNumParents() == 0) {
        tt_.erase(tt_iter);
      }
    }
  }

  return gc_queue_.empty();
}

}  // namespace lczero
