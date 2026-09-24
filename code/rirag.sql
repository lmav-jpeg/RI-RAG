/*
@author: Laurie MAVOUNGOU, JK AI CEO, lmavoungou@outlook.be*/

DROP DATABASE IF EXISTS RIRAG;

CREATE DATABASE RIRAG;
USE RIRAG;

CREATE TABLE file (
    file_id VARCHAR(50) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    semantic_summary TEXT NOT NULL,
    grade FLOAT NOT NULL,
    comments TEXT NOT NULL,
    CONSTRAINT file_pk PRIMARY KEY (file_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

INSERT INTO file(file_id, file_name, content, semantic_sumary, grade, comments)
VALUES
('DOC-001', 'consistency_guidelines.txt', 'The multi-round validation loop checks model responses against persistent evidence using database primary keys.', 'Defines multi-round contradiction detection and evidence grounding.', 9.5, 'Approved by lead engineer.'),
('DOC-002', 'agent_safety.txt', 'Frontier autonomous agents must maintain structural consistency to prevent cascading decision failures.', 'Autonomous agent safety and consistency rules.', 8.8, 'Needs minor revision on edge cases.');

/* Upcoming push for scaling tests. Need to make the payload heavier than the summary
 */