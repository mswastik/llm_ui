#!/usr/bin/env python3
"""
Complete migration script for ALL message formats to blocks structure.

This script handles:
1. OLD format: thinking column + simple tool_calls array
2. NEW format: tool_calls array with mixed types (thinking, content, tool_call)
3. Already migrated: messages that already have blocks in metadata

Usage:
    python migrate_all_messages_to_blocks.py [--preview]
    
Options:
    --preview   Show what would be migrated without making changes
"""

import sqlite3
import json
import sys
from datetime import datetime

DB_PATH = "llm_ui.db"


def migrate_all_messages(preview=False):
    """Migrate all messages to blocks structure."""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all messages
    cursor.execute("SELECT id, role, content, tool_calls, thinking, metadata FROM messages")
    rows = cursor.fetchall()
    
    print("=" * 70)
    print("Message Migration to Blocks Structure")
    print("=" * 70)
    print(f"Total messages: {len(rows)}")
    print()
    
    migrated_count = 0
    skipped_count = 0
    error_count = 0
    updates = []
    
    for row in rows:
        msg_id, role, content, tool_calls_json, thinking, metadata_json = row
        
        # Skip user messages
        if role != 'assistant':
            skipped_count += 1
            continue
        
        # Parse tool_calls
        try:
            tool_calls = json.loads(tool_calls_json) if tool_calls_json else []
        except:
            tool_calls = []
        
        # Parse metadata
        try:
            metadata = json.loads(metadata_json) if metadata_json else {}
        except:
            metadata = {}
        
        # Skip if already has blocks
        if metadata and 'blocks' in metadata and metadata['blocks']:
            skipped_count += 1
            continue
        
        blocks = []
        migration_type = None
        
        # Check if this is NEW format (tool_calls with type field)
        has_mixed_types = any(
            isinstance(tc, dict) and tc.get('type') in ('thinking', 'content', 'tool_call')
            for tc in tool_calls
        )
        
        if has_mixed_types:
            # NEW format: convert tool_calls with type field to blocks
            migration_type = "NEW format (mixed types in tool_calls)"
            for tc in tool_calls:
                tc_type = tc.get('type', 'tool_call')
                
                if tc_type == 'thinking':
                    blocks.append({
                        'type': 'thinking',
                        'content': tc.get('content', '')
                    })
                elif tc_type == 'content':
                    blocks.append({
                        'type': 'content',
                        'content': tc.get('content', '')
                    })
                elif tc_type == 'tool_call':
                    block = {
                        'type': 'tool_call',
                        'name': tc.get('name', 'unknown'),
                        'arguments': tc.get('arguments', tc.get('args', {})),
                        'status': tc.get('status', 'completed'),
                        'progress': tc.get('progress', 100),
                        'result': tc.get('result'),
                    }
                    # Copy optional fields
                    for field in ['sources', 'search_steps', 'search_terms', 'reasoning',
                                  'coverage_score', 'progress_history']:
                        if field in tc:
                            block[field] = tc[field]
                    blocks.append(block)
                else:
                    # Unknown type - treat as tool_call
                    blocks.append({
                        'type': 'tool_call',
                        'name': tc.get('name', 'unknown'),
                        'arguments': tc.get('arguments', tc.get('args', {})),
                        'status': tc.get('status', 'completed'),
                        'progress': tc.get('progress', 100),
                        'result': tc.get('result'),
                    })
        elif tool_calls or thinking:
            # OLD format: thinking column + simple tool_calls
            migration_type = "OLD format (thinking column + tool_calls)"
            
            # Add thinking from column
            if thinking and thinking.strip():
                blocks.append({
                    'type': 'thinking',
                    'content': thinking
                })
            
            # Add tool_calls
            for tc in tool_calls:
                if isinstance(tc, dict):
                    block = {
                        'type': 'tool_call',
                        'name': tc.get('name', 'unknown'),
                        'arguments': tc.get('arguments', tc.get('args', {})),
                        'status': tc.get('status', 'completed'),
                        'progress': tc.get('progress', 100),
                        'result': tc.get('result'),
                    }
                    # Copy optional fields
                    for field in ['sources', 'search_steps', 'search_terms', 'reasoning',
                                  'coverage_score', 'progress_history']:
                        if field in tc:
                            block[field] = tc[field]
                    blocks.append(block)
        
        # Add content if exists and we have other blocks
        if content and content.strip() and blocks:
            # Only add content block if there are other blocks (thinking/tool_call)
            blocks.append({
                'type': 'content',
                'content': content
            })
        
        # Store blocks if we have any
        if blocks:
            metadata['blocks'] = blocks
            
            if preview:
                block_types = [b['type'] for b in blocks[:5]]
                if len(blocks) > 5:
                    block_types.append(f'... ({len(blocks)} total)')
                print(f"  [PREVIEW] {msg_id[:8]}... - {migration_type}")
                print(f"            blocks: {block_types}")
            else:
                updates.append((json.dumps(metadata), msg_id))
                migrated_count += 1
                
                block_types = [b['type'] for b in blocks[:3]]
                if len(blocks) > 3:
                    block_types.append(f'... ({len(blocks)} total)')
                print(f"  ✓ {msg_id[:8]}... - {migration_type}")
                print(f"            blocks: {block_types}")
        else:
            skipped_count += 1
    
    # Execute updates if not preview
    if not preview and updates:
        cursor.executemany(
            "UPDATE messages SET metadata = ? WHERE id = ?",
            updates
        )
        conn.commit()
    
    conn.close()
    
    print()
    print("=" * 70)
    print("Migration Complete!")
    print("=" * 70)
    if preview:
        print(f"Would migrate: {migrated_count} messages")
        print(f"Would skip:    {skipped_count} messages")
    else:
        print(f"Migrated: {migrated_count} messages")
        print(f"Skipped:  {skipped_count} messages")
        print(f"Errors:   {error_count} messages")
    print("=" * 70)
    
    return migrated_count, skipped_count, error_count


if __name__ == '__main__':
    preview = '--preview' in sys.argv
    
    if preview:
        print("\n*** PREVIEW MODE - No changes will be made ***\n")
    
    migrate_all_messages(preview=preview)
