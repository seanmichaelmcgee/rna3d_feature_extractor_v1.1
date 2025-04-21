# Refactoring Remote Repository Strategy

This document outlines our strategy for managing multiple Git remotes during the refactoring process.

## Remote Repositories

We're using two remote repositories:

1. **Original Remote**: The original repository containing the pre-refactored code
   - Reference name: `origin`
   - URL: (original repository URL)

2. **Refactored Remote**: The repository for testing the refactored code
   - Reference name: `refactor`
   - URL: `git@github.com:seanmichaelmcgee/rna3d_feature_extractor_v2.1.0_test.git`

## Git Remote Management

Git can manage multiple remote repositories simultaneously. Here's how we're configuring it:

```bash
# View current remotes
git remote -v

# Add the refactored remote
git remote add refactor git@github.com:seanmichaelmcgee/rna3d_feature_extractor_v2.1.0_test.git

# Verify both remotes are configured
git remote -v
```

## Pushing to Different Remotes

When pushing commits, specify which remote to use:

```bash
# Push to original remote
git push origin main

# Push to refactored remote
git push refactor main
```

## Staging and Testing Plan

1. **Commit Current Work to Original Remote**:
   ```bash
   # Ensure all changes are committed
   git add .
   git commit -m "Complete integration phase of refactoring"
   
   # Push to original remote
   git push origin main
   ```

2. **Push to Refactored Remote for Testing**:
   ```bash
   # Push the same commits to the refactored remote
   git push refactor main
   ```

3. **Clone and Test the Refactored Repository**:
   ```bash
   # Clone the refactored repository to a new location
   git clone git@github.com:seanmichaelmcgee/rna3d_feature_extractor_v2.1.0_test.git /tmp/rna3d_test
   cd /tmp/rna3d_test
   
   # Set up the environment
   mamba activate rna3d-core
   
   # Run the tests
   python -m unittest discover
   ```

4. **Verification Checklist**:
   - [ ] All files cloned successfully
   - [ ] Directory structure is correct
   - [ ] All tests pass
   - [ ] Environment setup works
   - [ ] Feature extraction runs successfully

## Switching Active Remote

If you need to change which remote is the default for push/pull:

```bash
# Make refactor the default push destination for current branch
git branch --set-upstream-to=refactor/main main

# Or back to original
git branch --set-upstream-to=origin/main main
```

## Remote Status in Commit Messages

For clarity, include the target remote in the commit message when it matters:

```bash
git commit -m "[REFACTOR] Complete integration phase of refactoring"
```

## Rollback Procedure

If issues are found with the refactored repository, follow these steps:

1. Make necessary fixes in the original repository
2. Commit and push to original remote
3. Push to refactored remote
4. Re-test

## Final Transfer

Once all verification steps are complete and the refactoring is finalized:

1. Make final push to both remotes
2. Document the successful transfer
3. Update the team about the new primary repository location

## WARNING

Always verify which remote you're using before pushing sensitive or critical changes.
Always push to both remotes if the changes should be preserved in both repositories.