# Topology Get-Up Phase-11 Relative-Height Fix Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the discovered root cause that topology-getup progress terms use world-frame torso height, which saturates at reset on elevated terrain origins and breaks stall detection / progress shaping.

**Architecture:** Normalize topology-getup torso height against `env.scene.env_origins[:, 2]` when available. This keeps mocked tests simple while making real terrain runs use relative height above the local terrain patch.

**Tech Stack:** Python, pytest, torch
