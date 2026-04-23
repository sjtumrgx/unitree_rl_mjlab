# Topology Get-Up Phase-12 Support-Release Bonus Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Encourage the policy to leave the hand/knee support regime once it has achieved partial lift and facing-up progress.

**Architecture:** Add a one-shot `reduced_support_bonus` reward that pays when support-body contact count drops below a small threshold after enough torso lift / facing-up progress. Keep it local to topology-getup rewards/config and verify with a fresh bounded run.

**Tech Stack:** Python, pytest, torch
